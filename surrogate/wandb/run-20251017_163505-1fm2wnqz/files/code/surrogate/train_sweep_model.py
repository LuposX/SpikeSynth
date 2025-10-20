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
# import your datasets and SpikeSynth class

class SpikeSynth(L.LightningModule):
    def __init__(self, num_hidden, beta, optimizer_class, lr, batch_size, gamma):
        super().__init__()
        self.num_voltage_steps = 100
        self.num_voltage_steps = 100
        self.num_params = 6
        self.num_inputs = 1
        self.num_outputs = 100
        
        # Save hyperparameters for easy access
        self.save_hyperparameters()

        # Network layers
        self.linear0 = nn.Linear(self.num_inputs, self.hparams.num_hidden)
        self.lif0 = snn.Leaky(beta=self.hparams.beta)
        self.linear1 = nn.Linear(self.hparams.num_hidden, self.hparams.num_hidden)
        self.lif1 = snn.Leaky(beta=self.hparams.beta)
        self.linear2 = nn.Linear(self.hparams.num_hidden, self.hparams.num_hidden)
        self.lif2 = snn.Leaky(beta=self.hparams.beta)
        self.linear3 = nn.Linear(self.hparams.num_hidden + self.num_params, self.num_outputs)


    def forward(self, x):
        volt = x[:, :self.num_voltage_steps, :]
        params = x[:, self.num_voltage_steps:, 0]

        mem0 = self.lif0.init_leaky()
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        mem_rec = []
        for step in range(self.num_voltage_steps):
            x_step = self.linear0(volt[:, step, :])
            spk, mem0 = self.lif0(x_step, mem0)

            x_step = self.linear1(spk)
            spk, mem1 = self.lif1(x_step, mem1)
            
            x_step = self.linear2(spk)
            spk, mem2 = self.lif2(x_step, mem2)

            mem_rec.append(mem2)

        mem_final = torch.stack(mem_rec, dim=0)[-1]
        combined = torch.cat((mem_final, params), dim=1)
        out = self.linear3(combined)
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
        
        # Log the learning rate to W&B or progress bar
        self.log("lr", lr, prog_bar=True, on_step=False, on_epoch=True)


    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_class(self.parameters(), lr=self.hparams.lr)
        
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #    optimizer, mode='min'
        #)
        
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
        optimizer_class=torch.optim.Adam,
        beta=config.beta,
        lr=config.lr,
        num_hidden=config.num_hidden,
        batch_size=config.batch_size,
        gamma=config.gamma,
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
