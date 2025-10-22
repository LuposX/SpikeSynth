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

from RSNN import SpikeSynth

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
