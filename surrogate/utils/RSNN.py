#!/usr/bin/env python
    
import snntorch as snn
from snntorch._layers.bntt import BatchNormTT1d

import pytorch_lightning as L
import torch
import torch.nn as nn


class SpikeSynth(L.LightningModule):
    def __init__(self,
                 num_hidden_layers,
                 num_hidden,
                 beta,
                 optimizer_class,
                 lr,
                 batch_size,
                 dropout,
                 surrogate_gradient,
                 max_epochs,
                 temporal_skip,
                 layer_skip,
                 use_bntt,
                 scheduler_class,
                 scheduler_kwargs=None, 
                 bntt_time_steps=None,
                 train_dataset=None,
                 valid_dataset=None,
                 ):
        """
        Initializes the SpikeSynth model, a spiking neural network (SNN). 
        The model supports temporal and layer-wise skip connections,
        customizable surrogate gradients, and flexible optimizer/lr scheduling.

        Args:
            num_hidden_layers (int): 
                Number of recurrent Leaky Integrate-and-Fire (LIF) layers in the network.

            num_hidden (int): 
                Number of hidden neurons (units) in each LIF layer.

            beta (float): 
                Membrane potential decay constant for the LIF neurons (typically 0 < beta < 1).

            optimizer_class (type): 
                Optimizer class to use (e.g., `torch.optim.Adam`, `torch.optim.SGD`).

            lr (float): 
                Inital Learning rate for the optimizer.

            batch_size (int): 
                Batch size used for both training and validation DataLoaders.

            dropout (float):
                Value between [0, 1] if non zero, introduces a dropout layer.

            surrogate_gradient (torch.autograd.Function): 
                Surrogate gradient function for backpropagation through spikes
                (e.g., `snn.surrogate.atan()`).

            max_epochs (int): 
                Maximum number of training epochs. Used for scheduling when employing
                cosine annealing or similar schedulers.

            temporal_skip (int or -1): 
                Temporal skip interval in time steps for residual connections within
                each LIF layer. If `-1`, no temporal skip is applied.

            layer_skip (int or 0): 
                Number of layers to skip for inter-layer residual connections.
                For example, `layer_skip=1` adds a connection from each layer
                to the one immediately above it. If `0`, no layer skip is applied.

            use_bntt (bool):
                Whether to use 1D Batch Normalization Layer with length time_steps. 
                Uses Batch Normalisation Through Time (BNTT).

            bntt_time_steps (None or int):
                Batch Normalization Layer requries the amount of time steps that will be used i.e. sequecne length.

            train_dataset (torch.utils.data.Dataset, optional): 
                Dataset used for training. Required by `train_dataloader()`.

            valid_dataset (torch.utils.data.Dataset, optional): 
                Dataset used for validation. Required by `val_dataloader()`.

            scheduler_class (torch.optim.lr_scheduler):
                What Learning Rate Scheduler should be used for training.
        """
        super().__init__()
        
        self.num_params = 6
        self.num_inputs = 1
        self.num_outputs = self.num_inputs 
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.max_epochs = max_epochs
        self.surrogate_gradient = surrogate_gradient
        
        self.save_hyperparameters(ignore=["surrogate_gradient", "train_dataset", "valid_dataset"])

        self.norm = nn.LayerNorm(self.num_inputs + self.num_params)
        self.lif_layers = nn.ModuleList()

        self.temp_skip_projs = nn.ModuleList()
        
        # For layer-wise skip connections
        self.layer_skip_projs = nn.ModuleList()

        # Initialize BNTT modules if requested
        self.layer_bntt = nn.ModuleList()
        
        input_size = self.num_inputs + self.num_params
        for i in range(self.hparams.num_hidden_layers):
            self.lif_layers.append(
                snn.LeakyParallel(
                    input_size=input_size,
                    hidden_size=self.hparams.num_hidden,
                    beta=self.hparams.beta,
                    spike_grad=self.surrogate_gradient,
                    dropout=self.hparams.dropout
                )
            )
            
            # Temporal residual projection
            if input_size != self.hparams.num_hidden and self.hparams.temporal_skip != -1:
                self.temp_skip_projs.append(nn.Linear(input_size, self.hparams.num_hidden))
            else:
                self.temp_skip_projs.append(nn.Identity())
            
            # Layer-wise skip projection
            if self.hparams.layer_skip > 0 and i >= self.hparams.layer_skip:
                self.layer_skip_projs.append(nn.Linear(self.hparams.num_hidden, self.hparams.num_hidden))
            else:
                self.layer_skip_projs.append(None)

            if self.hparams.use_bntt:
                if self.hparams.bntt_time_steps is None:
                    raise ValueError("bntt_time_steps must be specified if use_bntt=True")
                else:
                    self.layer_bntt.append(BatchNormTT1d(self.hparams.num_hidden, self.hparams.bntt_time_steps))  
            else:
                self.layer_bntt.append(None)
                
            
            input_size = self.hparams.num_hidden

        self.output_layer = nn.Linear(self.hparams.num_hidden, self.num_outputs)
         
    def forward(self, x, params=None):
        # To log average amount of spikes
        self.spike_counts = []
        
        # Ensure x is 3D: (batch, seq_len, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() != 3:
            raise ValueError(f"x must be 2D or 3D, got {x.shape}")
    
        batch, seq_len, _ = x.shape  # channels is always 1
    
        # Handle static parameters
        if params is not None:
            if params.dim() < 2 or params.shape[1] < self.num_params:
                raise ValueError(f"params must have shape (batch, >= {self.num_params})")
            params_expanded = params.unsqueeze(2)
            x = torch.cat([params_expanded, x], dim=1)
        elif seq_len <= self.num_params:
            raise ValueError(f"x sequence too short to contain static params, got seq_len={seq_len}")
    
        # Separate static params and time series
        static_params = x[:, :self.num_params, 0]  # (batch, num_params)
        time_series = x[:, self.num_params:, 0]    # (batch, time_steps)
        time_steps = time_series.shape[1]
    
        # Repeat static params along time
        static_repeated = static_params.unsqueeze(1).repeat(1, time_steps, 1)  # (batch, time_steps, num_params)
        time_series_features = time_series.unsqueeze(2)  # (batch, time_steps, 1)
        x_transformed = torch.cat([static_repeated, time_series_features], dim=2)  # (batch, time_steps, num_inputs + num_params)
    
        # Permute to (T, B, F)
        x_seq = x_transformed.permute(1, 0, 2)  # (T, B, F_in)
        T, B, F = x_seq.shape
    
        skip_k = self.hparams.temporal_skip
    
        # Forward through LIF layers
        layer_outputs = [] 
        for i, lif in enumerate(self.lif_layers):
            lif_out = lif(x_seq)

            self.spike_counts.append(lif_out.mean())

            # Apply Batch Normalisation Through Time (BNTT)
            if self.hparams.use_bntt:
                lif_out_norm = []
                for t in range(lif_out.shape[0]):  # T dimension
                    lif_out_norm.append(self.layer_bntt[i][t](lif_out[t]))
                lif_out = torch.stack(lif_out_norm, dim=0)

            # Temporal skip connection
            if skip_k == -1 or skip_k >= T:
                x_seq = lif_out
            else:
                prev = torch.zeros_like(x_seq)
                prev[skip_k:] = x_seq[:-skip_k]
                prev_proj = self.temp_skip_projs[i](prev)
                x_seq = lif_out + prev_proj

            # Layer-wise skip connection
            if self.hparams.layer_skip > 0 and i >= self.hparams.layer_skip:
                skip_layer_out = layer_outputs[i - self.hparams.layer_skip]
                skip_proj = self.layer_skip_projs[i](skip_layer_out)
                x_seq = x_seq + skip_proj

            layer_outputs.append(x_seq)

        x_seq_b = x_seq.permute(1, 0, 2)
        out = self.output_layer(x_seq_b).squeeze()
        return out



    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        outputs = self(X_batch)
        loss = torch.nn.MSELoss()(outputs, y_batch.float())
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        spike_logs = {
            f"spikes/train_avg_layer_{i}": avg_spikes
            for i, avg_spikes in enumerate(self.spike_counts)
        }
        self.log_dict(spike_logs, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        outputs = self(X_batch)
        loss = torch.nn.MSELoss()(outputs, y_batch.float())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        opt = self.optimizers()
        lr = opt.param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
         optimizer = self.hparams.optimizer_class(self.parameters(), lr=self.hparams.lr)

         if self.hparams.scheduler_class is not None:
            # Add default T_max if scheduler is cosine and not provided
            if self.hparams.scheduler_class == torch.optim.lr_scheduler.CosineAnnealingLR:
                self.hparams.scheduler_kwargs.setdefault("T_max", self.hparams.max_epochs)
            scheduler = self.hparams.scheduler_class(optimizer, **self.hparams.scheduler_kwargs)

            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss",
            }
         else:
            return optimizer

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=8,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=4,
        )