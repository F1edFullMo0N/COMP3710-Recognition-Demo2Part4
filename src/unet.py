import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Conv -> BatchNorm -> ReLU
    Conv -> BatchNorm -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_classes=4
    ):
        super().__init__()

        # -------------------------------------------------
        # Encoder
        # -------------------------------------------------

        self.enc1 = DoubleConv(
            in_channels,
            32
        )

        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(
            32,
            64
        )

        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(
            64,
            128
        )

        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(
            128,
            256
        )

        self.pool4 = nn.MaxPool2d(2)

        # -------------------------------------------------
        # Bottleneck
        # -------------------------------------------------

        self.bottleneck = DoubleConv(
            256,
            512
        )

        # -------------------------------------------------
        # Decoder
        # -------------------------------------------------

        self.up4 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        # 256 decoder + 256 skip = 512
        self.dec4 = DoubleConv(
            512,
            256
        )

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            256,
            128
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            128,
            64
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            64,
            32
        )

        # -------------------------------------------------
        # Final pixel classifier
        # -------------------------------------------------

        self.output = nn.Conv2d(
            32,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        # [B, 32, 256, 256]

        e2 = self.enc2(
            self.pool1(e1)
        )
        # [B, 64, 128, 128]

        e3 = self.enc3(
            self.pool2(e2)
        )
        # [B, 128, 64, 64]

        e4 = self.enc4(
            self.pool3(e3)
        )
        # [B, 256, 32, 32]

        # Bottleneck
        b = self.bottleneck(
            self.pool4(e4)
        )
        # [B, 512, 16, 16]

        # -------------------------------------------------
        # Decoder + skip connections
        # -------------------------------------------------

        d4 = self.up4(b)
        # [B, 256, 32, 32]

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )
        # [B, 512, 32, 32]

        d4 = self.dec4(d4)
        # [B, 256, 32, 32]

        d3 = self.up3(d4)
        # [B, 128, 64, 64]

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        # [B, 64, 128, 128]

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        # [B, 32, 256, 256]

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        logits = self.output(d1)

        # [B, 4, 256, 256]
        return logits


if __name__ == "__main__":

    model = UNet(
        in_channels=1,
        num_classes=4
    )

    x = torch.randn(
        2,
        1,
        256,
        256
    )

    y = model(x)

    print("Input:", x.shape)
    print("Output:", y.shape)