#!/usr/bin/env python3
"""
train_rsnn.py

Example usage:

python train_rsnn.py --data ./data/dataset.ds --max-epochs 10 --batch-size 2048

Use --help to see all options.
"""

import argparse
import os
import logging
import torch
from torch.utils.data import TensorDataset

import snntorch as snn

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.model_summary import ModelSummary

from utils.RSNN import SpikeSynth

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def main(args):
    logger.info("Loading dataset from %s", args.data)
    data = torch.load(args.data)

    logger.info("Data keys: %s", list(data.keys()))
    logger.info("X_train shape: %s", tuple(data["X_train"].shape))
    logger.info("Y_train shape: %s", tuple(data["Y_train"].shape))

    # Extract tensors
    X_train, Y_train = data["X_train"], data["Y_train"]
    X_valid, Y_valid = data["X_valid"], data["Y_valid"]
    X_test, Y_test = data["X_test"], data["Y_test"]

    train_dataset = TensorDataset(X_train, Y_train)
    valid_dataset = TensorDataset(X_valid, Y_valid)
    test_dataset = TensorDataset(X_test, Y_test)

    # Logging directory setup
    script_dir = os.getcwd()
    logging_directory = os.path.join(script_dir, args.logging_directory)
    logging_directory = os.path.abspath(logging_directory)
    os.makedirs(logging_directory, exist_ok=True)
    os.environ["WANDB_DIR"] = logging_directory

    # Build surrogate gradient
    surrogate = None
    if args.surrogate == "atan":
        surrogate = snn.surrogate.atan()
    elif args.surrogate == "fast_sigmoid":
        surrogate = snn.surrogate.fast_sigmoid()
    else:
        logger.warning("Unknown surrogate '%s', falling back to atan()", args.surrogate)
        surrogate = snn.surrogate.atan()

    # Instantiate model
    model = SpikeSynth(
        optimizer_class=torch.optim.AdamW,
        beta=args.beta,
        lr=args.lr,
        num_hidden=args.num_hidden,
        batch_size=args.batch_size,
        num_hidden_layers=args.num_hidden_layers,
        dropout=args.dropout,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        max_epochs=args.max_epochs,
        surrogate_gradient=surrogate,
        temporal_skip=args.temporal_skip,
        layer_skip=args.layer_skip,
        use_bntt=args.use_bntt,
        bntt_time_steps=args.bntt_time_steps,
    )

    summary = ModelSummary(model, max_depth=1)
    num_params = summary.total_parameters
    trainable_params = summary.trainable_parameters
    logger.info("Model has %d trainable parameters (total %d)", trainable_params, num_params)

    # WandB logger
    wandb_logger = None
    if not args.no_wandb:
        logger.info("Initializing WandbLogger (project=%s name=%s) at %s", args.project_name, args.experiment_name, logging_directory)
        wandb_logger = WandbLogger(log_model=True, project=args.project_name, name=args.experiment_name, save_dir=logging_directory)
        try:
            wandb_logger.experiment.summary["trainable_parameters"] = trainable_params
            wandb_logger.experiment.summary["total_parameters"] = num_params
        except Exception as e:
            logger.warning("wandb_logger.experiment.summary() failed: %s", e)
        try:
            wandb_logger.watch(model)
        except Exception as e:
            logger.warning("wandb_logger.watch() failed: %s", e)
        try:
            wandb_logger.experiment.log_code(".", include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))
        except Exception as e:
            logger.warning("wandb_logger.experiment.log_code() failed: %s", e)
    else:
        logger.info("WandB disabled (--no-wandb).")

    # Checkpoint callback
    os.makedirs(args.checkpoint_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_path,
        filename=f"{args.experiment_name}spike_model-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=1,
        monitor=args.monitor,
        mode=args.monitor_mode,
    )

    accelerator = "gpu" if torch.cuda.is_available() and args.use_gpu_if_available else "cpu"
    logger.info("Using accelerator=%s (torch.cuda.is_available()=%s, use_gpu_if_available=%s)", accelerator, torch.cuda.is_available(), args.use_gpu_if_available)

    trainer_kwargs = dict(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        logger=wandb_logger if not args.no_wandb else None,
        callbacks=[checkpoint_callback],
        log_every_n_steps=args.log_every_n_steps,
    )

    trainer = Trainer(**trainer_kwargs)

    # Optionally compile model
    if args.torch_compile:
        try:
            logger.info("Attempting torch.compile(model)")
            model = torch.compile(model)
        except Exception as e:
            logger.warning("torch.compile failed: %s", e)

    # Train the model
    logger.info("Starting training for up to %d epochs", args.max_epochs)
    trainer.fit(model)

    # Finalize wandb
    if wandb_logger and not args.no_wandb:
        try:
            wandb_logger.finalize("success")
        except Exception as e:
            logger.warning("wandb_logger.finalize() failed: %s", e)

    logger.info("Best checkpoint saved at: %s", checkpoint_callback.best_model_path or "N/A")
    print("Best checkpoint saved at:", checkpoint_callback.best_model_path or "N/A")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SpikeSynth RSNN (converted from notebook)")

    parser.add_argument("--data", type=str, default="./data/dataset.ds", help="Path to dataset (torch file).")
    parser.add_argument("--max-epochs", type=int, default=10, help="Max number of training epochs.")
    parser.add_argument("--experiment-name", type=str, default="test", help="WandB experiment/run name.")
    parser.add_argument("--project-name", type=str, default="Spike-Synth-Surrogate", help="WandB project name.")
    parser.add_argument("--logging-directory", type=str, default=".temp", help="Local directory where logs/wandb files are stored.")
    parser.add_argument("--checkpoint-path", type=str, default="models/SRNN", help="Directory to save checkpoints.")
    parser.add_argument("--monitor", type=str, default="val_loss", help="Metric to monitor for checkpointing.")
    parser.add_argument("--monitor-mode", dest="monitor_mode", choices=["min", "max"], default="min", help="Monitor mode for checkpointing/early stopping.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging.")

    # Model/training hyperparameters
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size (passed to SpikeSynth).")
    parser.add_argument("--num-hidden", type=int, default=256, help="Number of hidden units.")
    parser.add_argument("--num-hidden-layers", type=int, default=4, help="Number of hidden layers.")
    parser.add_argument("--beta", type=float, default=0.9, help="Beta (optimizer momentum-like).")
    parser.add_argument("--temporal-skip", type=int, default=-1, help="Temporal skip value (or None).")
    parser.add_argument("--layer-skip", type=int, default=2, help="Layer skip value.")
    parser.add_argument("--surrogate", type=str, default="atan", help="Surrogate gradient to use (e.g. 'atan').")
    parser.add_argument("--dropout", type=float, default=0, help="Introduces a dropout layer for each LIF layer.")
    parser.add_argument("--torch-compile", action="store_true", help="Attempt torch.compile(model) before training.")
    parser.add_argument("--use-gpu-if-available", action="store_true", help="Use GPU if available (default: off).")
    parser.add_argument("--use-bntt", type=str2bool, default=False, help="Whetehr to use Batchnorm or not.")
    parser.add_argument("--bntt-time-steps", type=int, default=100, help="Batchnorm needs to know the sequence length beforehand.")
    parser.add_argument("--log-every-n-steps", type=int, default=10, help="Logging frequency (trainer.log_every_n_steps)")

    args = parser.parse_args()

    main(args)