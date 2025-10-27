#!/usr/bin/env python3
"""
Creates a wandb sweep from a YAML config and runs a wandb.agent
which executes the training function in-process.
This basically does a hyperparameetr search over possbile valeus as defiend in the sweep.yaml.

Usage:
    python 3_hyperparameter_search_rsnn.py --sweep-config sweep_rsnn.yaml --project test
"""

import argparse
import yaml
import inspect
import os
import sys

import wandb

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import snntorch as snn

from utils.RSNN import SpikeSynth


def build_surrogate(surr_name, config):
    """Dynamically build a surrogate gradient callable from its name and sweep config."""
    if not hasattr(snn.surrogate, surr_name):
        raise ValueError(f"surrogate '{surr_name}' not found in snn.surrogate")
    surr_fn = getattr(snn.surrogate, surr_name)
    sig = inspect.signature(surr_fn)

    # Gather kwargs only for parameters the surrogate accepts and that exist in config
    kwargs = {}
    for pname in sig.parameters:
        if hasattr(config, pname):
            kwargs[pname] = getattr(config, pname)

    print(f"[build_surrogate] Building surrogate '{surr_name}' with kwargs: {kwargs}")
    return surr_fn(**kwargs)


def training_run():
    """
    The function to be passed to wandb.agent. It will be called repeatedly by wandb.agent.
    Each call should create a wandb run (wandb.init()) and perform a single training run.
    """
    # Start a wandb run; wandb.agent will set the sweep config values into wandb.config
    run = wandb.init()
    try:
        config = wandb.config

        # Load dataset - same path as your original script
        data_path = "./data/dataset.ds"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        data = torch.load(data_path)
        X_train, Y_train = data['X_train'], data['Y_train']
        X_valid, Y_valid = data['X_valid'], data['Y_valid']
        X_test, Y_test = data['X_test'], data['Y_test']

        train_dataset = TensorDataset(X_train, Y_train)
        valid_dataset = TensorDataset(X_valid, Y_valid)
        test_dataset  = TensorDataset(X_test, Y_test)

        # Build surrogate gradient
        spike_grad = build_surrogate(config.surrogate_gradient, config)

        # Build model instance (mirror your original kwargs)
        model = SpikeSynth(
            optimizer_class=getattr(torch.optim, config.optimizer_class),
            beta=config.beta,
            lr=config.lr,
            num_hidden=config.num_hidden,
            batch_size=config.batch_size,
            gamma=config.gamma,
            surrogate_gradient=spike_grad,
            num_hidden_layers=config.num_hidden_layers,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            max_epochs=config.epochs,
            temporal_skip=config.temporal_skip,
            layer_skip=config.layer_skip
        )

        # Setup WandbLogger for PyTorch Lightning
        wandb_logger = WandbLogger(
            log_model=False,
            project=wandb.run.project or "Spike-Synth-rsnn",
            name=f"sweep_run_{wandb.run.id}"
        )

        # Optionally watch model (be careful about large models / autolog volume)
        try:
            wandb_logger.watch(model)
        except Exception as e:
            print(f"[warning] wandb_logger.watch failed: {e}")

        # Define checkpoint callback
        checkpoint_dir = os.path.join(
            os.getenv("WANDB_DIR", "."),
            "checkpoints",
            wandb.run.project,
            wandb.run.sweep_id or "manual",
            wandb.run.id
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="epoch{epoch:02d}-val_loss{val_loss:.2f}",
            save_top_k=1,              # keep best 3 models
            monitor="val_loss",        # assumes your model logs 'val_loss'
            mode="min",
            save_last=True,
        )
        
        # Choose accelerator
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        trainer = Trainer(
            max_epochs=config.epochs,
            logger=wandb_logger,
            accelerator=accelerator,
            callbacks=[checkpoint_callback]
        )

        # Fit
        trainer.fit(model)

    finally:
        # Ensure the run is closed
        wandb.finish()


def create_and_run_sweep(sweep_config_path: str, project: str, logging_directory : str, entity: str = None):
    # Set and Create logging directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logging_directory = os.path.join(script_dir, logging_directory)
    logging_directory = os.path.abspath(logging_directory)
    os.makedirs(logging_directory, exist_ok=True)
    os.environ["WANDB_DIR"] = logging_directory
    
    # Load sweep YAML
    with open(sweep_config_path, "r") as f:
        sweep_config = yaml.safe_load(f)

    # Create sweep (returns sweep id)
    print(f"[sweep] Creating sweep for project='{project}', entity='{entity or wandb.Api().default_entity}'")
    sweep_id = wandb.sweep(sweep=sweep_config, project=project, entity=entity)
    print(f"[sweep] Created sweep id: {sweep_id}")

    # Determine run cap (count) for agent
    count = None
    # common keys used: 'run_cap' in your YAML; also check nested
    if isinstance(sweep_config, dict):
        if "run_cap" in sweep_config:
            count = sweep_config["run_cap"]
        elif "run" in sweep_config and isinstance(sweep_config["run"], dict):
            count = sweep_config["run"].get("run_cap", None)

    print(f"[sweep] Starting wandb.agent with count={count} (None means run until stopped).")
    # Run the agent programmatically, using training_run as the function to execute for each
    wandb.agent(project + "/" + sweep_id if "/" not in str(sweep_id) else sweep_id,
                function=training_run,
                count=count)


def main():
    parser = argparse.ArgumentParser(description="Create and run a wandb sweep (combined).")
    parser.add_argument("--sweep-config", "-c", default="sweep_hyperparameter_rsnn.yaml",
                        help="Path to sweep YAML config (default: sweep_hyperparameter_rsnn.yaml)")
    parser.add_argument("--project", "-p", default="test", help="W&B project name (default: test)")
    parser.add_argument("--entity", "-e", default=None, help="W&B entity (user or team), optional")
    parser.add_argument("--logging-directory", "-l", default=".temp", help="Local path of the directory we log to.")
    args = parser.parse_args()

    # Make sure no stray run is left open
    try:
        wandb.finish()
    except Exception:
        pass

    create_and_run_sweep(args.sweep_config, args.project, args.logging_directory, args.entity)


if __name__ == "__main__":
    main()
