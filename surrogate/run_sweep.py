import wandb
import os
import subprocess
import yaml

# Path to your sweep config
SWEEP_CONFIG_PATH = "sweep_rsnn.yaml"
PROJECT_NAME = "test"

# Load the YAML file into a dict
with open(SWEEP_CONFIG_PATH, "r") as f:
    sweep_config = yaml.safe_load(f)

# Create the sweep
sweep_id = wandb.sweep(
    sweep=sweep_config,
    project=PROJECT_NAME
)

print(f"Created sweep with ID: {sweep_id}")

entity = wandb.Api().default_entity

# Run the agent
subprocess.run(["wandb", "agent", f"{entity}/{PROJECT_NAME}/{sweep_id}"])