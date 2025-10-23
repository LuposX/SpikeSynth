#!/usr/bin/env python
    
import snntorch as snn
import pytorch_lightning as L
import torch
import torch.nn as nn


class SpikeSynth(L.LightningModule):
    def __init__(self, num_hidden_layers, num_hidden, beta, optimizer_class, lr, batch_size, gamma, surrogate_gradient, train_dataset, valid_dataset, max_epochs):
        super().__init__()
        self.num_params = 6
        self.num_inputs = 1
        self.num_outputs = self.num_inputs 
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.max_epochs = max_epochs
        self.surrogate_gradient = surrogate_gradient

        self.save_hyperparameters(ignore=["surrogate_gradient"])

        self.lif_layers = nn.ModuleList()

        input_size = self.num_inputs + self.num_params
        for _ in range(self.hparams.num_hidden_layers):
            self.lif_layers.append(
                snn.LeakyParallel(
                    input_size=input_size,
                    hidden_size=self.hparams.num_hidden,
                    beta=self.hparams.beta,
                    spike_grad=self.surrogate_gradient
                )
            )
            # After first layer, input_size = hidden_size
            input_size = self.hparams.num_hidden

        self.norm = nn.LayerNorm(self.hparams.num_hidden)
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
        if params.dim() < 2:
          raise ValueError(f"params must have at least 2 dimensions (batch size x static parameters), got {params.shape}")
        if params.shape[1] < self.num_params:
          raise ValueError(f"params must have at least {self.num_params} parameters per batch, got {params.shape[1]}")

        # Expand params across the time dimension
        params_expanded = params.unsqueeze(2)
        x = torch.cat([x, params_expanded], dim=1)

      else:
        # Assume last 6 timesteps of x are static params if not provided
        if seq_len <= self.num_params:
            raise ValueError(f"x sequence too short to contain static params, got seq_len={seq_len}")
        x_features = x  # dynamic + static already included

      static_params = x[:, :self.num_params, 0]  # shape: (batch_size, 6)
      time_series = x[:, self.num_params:, 0]    # shape: (batch_size, time_series_length)
      time_steps = time_series.shape[1]

      # Repeat static parameters for each time step
      static_repeated = static_params.unsqueeze(1).repeat(1, time_steps, 1)  # shape: (batch_size, time_steps, 6)

      # Add the current time step value as the last feature
      time_series_features = time_series.unsqueeze(2)  # shape: (batch_size, time_steps, 1)

      # Concatenate static + time series
      x_transformed = torch.cat([static_repeated, time_series_features], dim=2)  # shape: (batch_size, time_steps, 7)
      x_seq = x_transformed.permute(1, 0, 2)  # shape: (time_steps, batch_size, 7)

      # Forward through all LIF layers
      for lif in self.lif_layers:
          x_seq = lif(x_seq)

      # Swap back to batch-first
      x_seq_b = x_seq.permute(1, 0, 2)  # (batch, seq_len, hidden_size)

      x_seq_b = self.norm(x_seq_b)

      out = self.output_layer(x_seq_b).squeeze()  # (batch, seq_len, num_outputs)

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