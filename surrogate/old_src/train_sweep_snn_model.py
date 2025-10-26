#!/usr/bin/env python

## Important
#
# Right now, the model predicts a static output vector (of size 100) given a temporal input (100 timesteps of voltages + 6 params).
# It’s not predicting the next voltages over time, it’s mapping a temporal pattern to an output vector. It's not autoregressive.
# 
# Future devlopment:
# - Goal: 
# $$
# v_{t+1} = f(v_{t}, v_{t-1}, \ldots, v_{t-w}, \text{params})
# $$
# - Use https://snntorch.readthedocs.io/en/latest/snn.neurons_rleaky.html
# - Feed the model a window of previous timesteps, and have it predict the next one:
# ```python
# for t in range(window, total_steps - 1):
#     input = volt[:, t - window:t, :]
#     target = volt[:, t + 1, :]
# ```
# 
# Questions:
# - Do I have the 6 static parameter during inference?
# - the mdoel shoult be autoregressive right? I.e. given a tiem sequence predict the next "token".

    
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
        self.num_voltage_steps = 100
        self.num_params = 6
        self.num_inputs = 1
        self.num_outputs = 100

        self.save_hyperparameters()

        # Build dynamic hidden layers
        self.hidden_layers = nn.ModuleList()
        self.lif_layers = nn.ModuleList()

        # First input layer
        self.hidden_layers.append(nn.Linear(self.num_inputs, self.hparams.num_hidden))
        self.lif_layers.append(snn.Leaky(beta=self.hparams.beta))

        # Additional hidden layers
        for _ in range(1, self.hparams.num_hidden_layers):
            self.hidden_layers.append(nn.Linear(self.hparams.num_hidden, self.hparams.num_hidden))
            self.lif_layers.append(snn.Leaky(beta=self.hparams.beta))

        # Output layer takes final hidden layer output + parameters
        self.output_layer = nn.Linear(self.hparams.num_hidden + self.num_params, self.num_outputs)

    def forward(self, x):
        volt = x[:, :self.num_voltage_steps, :]
        params = x[:, self.num_voltage_steps:, 0]

        # Initialize membrane potentials for each LIF layer
        mem_states = [lif.init_leaky() for lif in self.lif_layers]

        mem_rec = []
        for step in range(self.num_voltage_steps):
            x_step = volt[:, step, :]

            for i, (linear, lif) in enumerate(zip(self.hidden_layers, self.lif_layers)):
                x_step = linear(x_step)
                spk, mem_states[i] = lif(x_step, mem_states[i])
                x_step = spk  # feed spikes forward

            mem_rec.append(mem_states[-1])  # record last layer’s membrane

        mem_final = torch.stack(mem_rec, dim=0)[-1]
        combined = torch.cat((mem_final, params), dim=1)
        out = self.output_layer(combined)
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
        project="Spike-Synth",
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
