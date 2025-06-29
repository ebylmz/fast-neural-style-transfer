import torch.nn as nn

class ConvLayer(nn.Module):
    """
    A Convolution -> InstanceNorm -> (optional) ReLU block
    Uses reflection padding to avoid edge artifacts.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, relu=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels, affine=True)
        ]
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Residual Block: Conv-IN-ReLU -> Conv-IN -> skip connection
    No ReLU after the skip connection, following He et al.
    """
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvLayer(channels, channels, kernel_size=3, stride=1, relu=True),
            ConvLayer(channels, channels, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return x + self.block(x)


class UpsampleConvLayer(nn.Module):
    """
    Nearest-neighbor upsampling followed by Conv2D and InstanceNorm + ReLU.
    Avoids checkerboard artifacts seen in transposed convolutions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, upsample=None):
        super().__init__()
        layers = []
        if upsample:
            layers.append(nn.Upsample(scale_factor=upsample, mode='nearest'))
        layers.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TransformNet(nn.Module):
    """
    Feed-forward network for fast style transfer
    """
    def __init__(self):
        super().__init__()

        # Downsampling
        self.downsampling = nn.Sequential(
            ConvLayer(3, 32, kernel_size=9, stride=1),
            ConvLayer(32, 64, kernel_size=3, stride=2),
            ConvLayer(64, 128, kernel_size=3, stride=2)
        )

        # Residual blocks
        self.residuals = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )

        # Upsampling
        self.upsampling = nn.Sequential(
            UpsampleConvLayer(128, 64, kernel_size=3, upsample=2),
            UpsampleConvLayer(64, 32, kernel_size=3, upsample=2),
            nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4, padding_mode='reflect')
            # No Tanh here
        )

    def forward(self, x):
        x = self.downsampling(x)
        x = self.residuals(x)
        x = self.upsampling(x)
        return x
