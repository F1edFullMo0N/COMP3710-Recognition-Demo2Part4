import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()

        self.latent_dim = latent_dim

        # Encoder

        self.encoder = nn.Sequential(

            # [B, 1, 256, 256]
            nn.Conv2d(
                1, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            # [B, 32, 128, 128]
            nn.Conv2d(
                32, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # [B, 64, 64, 64]
            nn.Conv2d(
                64, 128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # [B, 128, 32, 32]
            nn.Conv2d(
                128, 256,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU()

            # Output: [B, 256, 16, 16]
        )

        self.flatten_dim = 256 * 16 * 16

        # VAE produces two latent parameters
        self.fc_mu = nn.Linear(
            self.flatten_dim,
            latent_dim
        )

        self.fc_logvar = nn.Linear(
            self.flatten_dim,
            latent_dim
        )

        # Convert latent vector back to feature map
        self.decoder_input = nn.Linear(
            latent_dim,
            self.flatten_dim
        )

        # Decoder

        self.decoder = nn.Sequential(

            # [B, 256, 16, 16]
            nn.ConvTranspose2d(
                256, 128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # [B, 128, 32, 32]
            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # [B, 64, 64, 64]
            nn.ConvTranspose2d(
                64, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # [B, 32, 128, 128]
            nn.ConvTranspose2d(
                32, 1,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            # Output pixels in [0, 1]
            nn.Sigmoid()

            # Output: [B, 1, 256, 256]
        )

    def encode(self, x):
        x = self.encoder(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        # logvar = log(sigma^2)
        # sigma = exp(0.5 * logvar)

        std = torch.exp(0.5 * logvar)

        # epsilon ~ N(0, 1)
        eps = torch.randn_like(std)

        # z = mu + sigma * epsilon
        z = mu + eps * std

        return z

    def decode(self, z):
        x = self.decoder_input(z)

        x = x.view(
            -1,
            256,
            16,
            16
        )

        x = self.decoder(x)

        return x

    def forward(self, x):
        mu, logvar = self.encode(x)

        z = self.reparameterize(
            mu,
            logvar
        )

        reconstruction = self.decode(z)

        return reconstruction, mu, logvar