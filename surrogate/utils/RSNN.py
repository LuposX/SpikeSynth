#!/usr/bin/env python
    
import snntorch as snn
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
                 gamma,
                 surrogate_gradient,
                 max_epochs,
                 temporal_skip,
                 layer_skip,
                 train_dataset=None,
                 valid_dataset=None,
                 ):
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
        self.residual_alphas = nn.ParameterList(
            [nn.Parameter(torch.ones(1)) for _ in range(self.hparams.num_hidden_layers)]
        )
        self.residual_projs = nn.ModuleList()
        
        # For layer-wise skip connections
        self.layer_skip_alphas = nn.ParameterList()
        self.layer_skip_projs = nn.ModuleList()
        
        input_size = self.num_inputs + self.num_params
        for i in range(self.hparams.num_hidden_layers):
            self.lif_layers.append(
                snn.LeakyParallel(
                    input_size=input_size,
                    hidden_size=self.hparams.num_hidden,
                    beta=self.hparams.beta,
                    spike_grad=self.surrogate_gradient
                )
            )
            
            # Temporal residual projection
            if input_size != self.hparams.num_hidden:
                self.residual_projs.append(nn.Linear(input_size, self.hparams.num_hidden))
            else:
                self.residual_projs.append(nn.Identity())
            
            # Layer-wise skip projection
            if self.hparams.layer_skip > 0 and i >= self.hparams.layer_skip:
                self.layer_skip_alphas.append(nn.Parameter(torch.ones(1)))
                self.layer_skip_projs.append(nn.Linear(self.hparams.num_hidden, self.hparams.num_hidden))
            else:
                self.layer_skip_alphas.append(None)
                self.layer_skip_projs.append(None)
            
            input_size = self.hparams.num_hidden

        self.output_layer = nn.Linear(self.hparams.num_hidden, self.num_outputs)
         
    def forward(self, x, params=None):
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

            # Temporal skip connection
            if skip_k is None or skip_k >= T:
                x_seq = lif_out
            else:
                prev = torch.zeros_like(x_seq)
                prev[skip_k:] = x_seq[:-skip_k]
                prev_proj = self.residual_projs[i](prev)
                x_seq = lif_out + self.residual_alphas[i] * prev_proj

            # Layer-wise skip connection
            if self.hparams.layer_skip > 0 and i >= self.hparams.layer_skip:
                skip_layer_out = layer_outputs[i - self.hparams.layer_skip]
                skip_proj = self.layer_skip_projs[i](skip_layer_out)
                x_seq = x_seq + self.layer_skip_alphas[i] * skip_proj

            layer_outputs.append(x_seq)

        x_seq_b = x_seq.permute(1, 0, 2)
        out = self.output_layer(x_seq_b).squeeze()
        return out



    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        outputs = self(X_batch)
        loss = torch.nn.MSELoss()(outputs, y_batch.float())
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
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
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #    optimizer, gamma=self.hparams.gamma
        #)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

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