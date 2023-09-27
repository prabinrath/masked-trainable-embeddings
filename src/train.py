"""
train.py

Train a Latent Actions Model (Conditional AutoEncoder; CAE) on Real-Robot Kinesthetic Demonstrations. Packages
code for defining PyTorch-Lightning Modules for Latent Action Models, and for performing data augmentation & training.

Additionally saves model checkpoints and logs statistics.
"""
import sys, os

sys.path.append("/home/local/ASUAD/opatil3/src/SCLMaps/")
from argparse import Namespace
from pytorch_lightning.callbacks import ModelCheckpoint
from tap import Tap
from torch.utils.data import DataLoader
from typing import List

from src.lightning_logging import MetricLogger
from src.models import CAE
from src.preprocessing import get_dataset

import numpy as np
import os
import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision("medium")


class ArgumentParser(Tap):
    model = "cae"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint: str = "checkpoints/"                            # Path to Checkpoint Directory

    # Model Parameters
    latent_dim: int = 1                                         # Dimensionality of Latent Space (match input)
    state_dim: int = 1                                          # Dimensionality of Robot State (7-DoF Joint Angles)

    # CAE Model Parameters
    hidden: int = 30                                            # Size of AutoEncoder Hidden Layer (Dylan Magic)

    # GPUs
    gpus: int = 0                                               # Number of GPUs to run with (defaults to cpu)

    # Training Parameters
    epochs: int = 100                                           # Number of training epochs to run
    bsz: int = 64                                               # Batch Size for training
    lr: float = 0.01                                            # Learning Rate for training
    lr_step_size: int = 400                                     # How many epochs to run before LR decay
    lr_gamma: float = 0.1                                       # Learning Rate Gamma (decay rate)

    # Train-Val/Augmentation/Noise Parameters
    val_split: float = 0.1                                      # Percentage of Data to use as Validation
    noise_std: float = 0.01                                     # Standard Deviation for Gaussian to draw noise from
    window: int = 10                                            # Window-Shift Augmentation to apply to demo states

    # Random Seed
    seed: int = 21                                              # Random Seed (for Reproducibility)


def train():
    # Parse Arguments --> Convert from Namespace --> Dict --> Namespace because of weird WandB Bug
    print("[*] Starting up...")
    args = Namespace(**ArgumentParser().parse_args().as_dict())
    print(
        '\t[*] "Does the walker choose the path, or the path the walker?" (Garth Nix - Sabriel)\n'
    )

    # Create Run Name
    run_name = (
        f"{args.model}-z={args.latent_dim}-w={args.window}"
        f"-n={args.noise_std:.2f}-h={args.hidden}-ep={args.epochs}-x{args.seed}"
    )

    # Set Randomness + Device
    print(f"[*] Setting Random Seed to {args.seed}!\n")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("[*] Creating Dummy Training and Validation Datasets...\n")
    train_dataset, val_dataset = get_dataset(1000)

    # Initialize DataLoaders
    train_loader = (
        DataLoader(
            dataset=train_dataset, batch_size=args.bsz, shuffle=True, num_workers=24
        ),
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=args.bsz, shuffle=False, num_workers=24
    )

    # Create Model
    print("[*] Initializing Latent Actions Model...\n")
    if args.model == "cae":
        nn = CAE(args)

    # Create Trainer
    print("[*] Training...\n")
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint, "runs", run_name, run_name),
        filename="{train_loss:.2f}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    logger = MetricLogger(name=run_name, save_dir=args.checkpoint)
    trainer = pl.Trainer(
        default_root_dir=args.checkpoint,
        max_epochs=args.epochs,
        num_nodes=args.gpus,
        logger=logger,
        callbacks=checkpoint_callback,
    )

    # Fit
    trainer.fit(nn, train_loader, val_loader)


if __name__ == "__main__":
    train()
