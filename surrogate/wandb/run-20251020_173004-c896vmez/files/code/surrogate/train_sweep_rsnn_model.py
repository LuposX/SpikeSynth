#!/usr/bin/env python
    
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as L

from pytorch_lightning import Trainer
import wandb

import os
import wandb

import torch
import torch.nn as nn
import snntorch as snn

class SpikeSynth(L.LightningModule):
    def __init__(self, num_hidden_layers, num_hidden, beta, optimizer_class, lr, batch_size, gamma):
        super().__init__()
        self.num_params = 6
        self.num_inputs = 1
        self.num_outputs = 100

        self.save_hyperparameters()

        self.lif_layers = nn.ModuleList()

        input_size = self.num_inputs + self.num_params
        for _ in range(self.hparams.num_hidden_layers):
            self.lif_layers.append(
                snn.LeakyParallel(
                    input_size=input_size,
                    hidden_size=self.hparams.num_hidden,
                    beta=self.hparams.beta
                )
            )
            # After first layer, input_size = hidden_size
            input_size = self.hparams.num_hidden

        self.output_layer = nn.Linear(self.hparams.num_hidden, self.num_inputs)

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
          x_seq = lif(x_seq)  # (L, batch, hidden_size)

      # Take last timestep
      last = x_seq[-1]  # (batch, hidden_size)
      out = self.output_layer(last)  # (batch, num_outputs)

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
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.hparams.gamma
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=8,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=4,
        )



if __name__ == "__main__":
    # Our data in in the shape: trainings samples(28k) * number of time steps (100 + 6) * time dimension(1)
    # The time steps is voltage over time
    data = torch.load(f'./data/dataset.ds')
    # Extract tensors
    X_train, Y_train = data['X_train'], data['Y_train']
    X_valid, Y_valid = data['X_valid'], data['Y_valid']
    X_test, Y_test = data['X_test'], data['Y_test']

    train_dataset = TensorDataset(X_train, Y_train)
    valid_dataset = TensorDataset(X_valid, Y_valid)
    test_dataset  = TensorDataset(X_test, Y_test)


    # Initialize wandb run
    wandb.init()
    config = wandb.config


    # Create your model using hyperparameters from config
    model = SpikeSynth(
        optimizer_class=getattr(torch.optim, wandb.config.optimizer_class),
        beta=config.beta,
        lr=config.lr,
        num_hidden=config.num_hidden,
        batch_size=config.batch_size,
        gamma=config.gamma,
        num_hidden_layers=config.num_hidden_layers
    )

    # Use WandbLogger
    wandb_logger = WandbLogger(
        log_model=False,
        project="Spike-Synth-rsnn",
        name=f"sweep_run_{wandb.run.id}"
    )
    wandb_logger.watch(model)

    trainer = Trainer(
        max_epochs=config.epochs,
        logger=wandb_logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )

    # Train the model
    trainer.fit(model)
    wandb.finish()
