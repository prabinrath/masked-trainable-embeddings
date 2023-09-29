"""
cae.py

PyTorch-Lightning Module Definition for the CAE Latent Actions Model.
"""
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CAE(pl.LightningModule):
    def __init__(self, hparams):
        super(CAE, self).__init__()

        # Save Hyper-Parameters
        self.cae_hparams = hparams

        # Build Model
        self.build_model()

    def build_model(self):
        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(
                self.cae_hparams.state_dim + self.cae_hparams.state_dim,
                self.cae_hparams.hidden,
            ),
            nn.Tanh(),
            nn.Linear(self.cae_hparams.hidden, self.cae_hparams.hidden),
            nn.Tanh(),
            nn.Linear(self.cae_hparams.hidden, self.cae_hparams.latent_dim),
        )

        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(
                self.cae_hparams.state_dim + self.cae_hparams.state_dim,
                self.cae_hparams.hidden,
            ),
            nn.Tanh(),
            nn.Linear(self.cae_hparams.hidden, self.cae_hparams.hidden),
            nn.Tanh(),
            nn.Linear(self.cae_hparams.hidden, self.cae_hparams.state_dim),
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.cae_hparams.lr)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.cae_hparams.lr_step_size,
            gamma=self.cae_hparams.lr_gamma,
        )

        return [optimizer], [scheduler]

    def forward(self, s, a):
        # Create Input to Encoder --> State, Action
        x = torch.cat((s, a), 1)
        z = self.enc(x)

        # Create Input to Decoder --> State, Latent Action
        x = torch.cat((s, z), 1)

        # Return Predicted Action
        return self.dec(x)

    def training_step(self, batch, batch_idx):
        # Extract Batch
        state, action = batch[0]["state"].type(torch.FloatTensor).to(
            self.cae_hparams.device
        ), batch[0]["action"].type(torch.FloatTensor).to(self.cae_hparams.device)

        # Get Predicted Action
        predicted_action = self.forward(state.unsqueeze(-1), action.unsqueeze(-1))

        # Measure MSE Loss
        loss = F.mse_loss(predicted_action, action.unsqueeze(-1))

        # Log Loss
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        # Extract Batch
        state, action = batch["state"].type(torch.FloatTensor).to(
            self.cae_hparams.device
        ), batch["action"].type(torch.FloatTensor).to(self.cae_hparams.device)

        # Get Predicted Action
        predicted_action = self.forward(state.unsqueeze(-1), action.unsqueeze(-1))

        # Measure MSE Loss
        loss = F.mse_loss(predicted_action, action.unsqueeze(-1))

        # Log Loss
        self.log("val_loss", loss, prog_bar=True)
