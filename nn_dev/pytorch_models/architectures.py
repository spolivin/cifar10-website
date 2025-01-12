import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Builds a number of convolutional blocks."""

    def __init__(self, encoder_channels: tuple[int]) -> None:
        """Initializes a class instance.

        Args:
            encoder_channels (tuple[int]): Convolution channels.
        """
        super().__init__()
        self.conv_blocks = nn.Sequential(
            *[
                self.conv_block(in_channel, out_channel)
                for in_channel, out_channel in zip(
                    encoder_channels, encoder_channels[1:]
                )
            ]
        )

    def conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Creates a convolution block of the Encoder.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            nn.Sequential: Convolution block of sequentially connected layers.
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Makes a forward pass.

        Args:
            x (torch.Tensor): Input tensor for Encoder.

        Returns:
            torch.Tensor: Output tensor of Encoder.
        """
        return self.conv_blocks(x)


class Decoder(nn.Module):
    """Builds a number of feedforward blocks."""

    def __init__(self, decoder_features: tuple[int], num_labels: int) -> None:
        """Initializes a class instance.

        Args:
            decoder_features (tuple[int]): Channels in the Decoder.
            num_labels (int): Number of output labels.
        """
        super().__init__()
        self.dec_blocks = nn.Sequential(
            *[
                self.dec_block(in_feature, out_feature)
                for in_feature, out_feature in zip(
                    decoder_features, decoder_features[1:]
                )
            ]
        )
        self.dropout = nn.Dropout(p=0.5)
        self.last = nn.Linear(decoder_features[-1], num_labels)

    def dec_block(self, in_features: int, out_features: int) -> nn.Sequential:
        """Creates a decoder block of the Decoder.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.

        Returns:
            nn.Sequential: Decoder block of sequentially connected layers.
        """
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Makes a forward pass.

        Args:
            x (torch.Tensor): Input tensor for Decoder.

        Returns:
            torch.Tensor: Output tensor of Decoder.
        """
        x = self.dec_blocks(x)
        x = self.dropout(x)
        x = self.last(x)

        return x


class CNN(nn.Module):
    """Joins Encoder with Decoder to form a CNN."""

    def __init__(
        self,
        in_channels: int,
        encoder_channels: tuple[int],
        decoder_features: tuple[int],
        num_labels: int,
    ) -> None:
        """Initializes a class instance.

        Args:
            in_channels (int): Number of input channels.
            encoder_channels (tuple[int]): Encoder channels.
            decoder_features (tuple[int]): Decoder channels.
            num_labels (int): Number of ouput labels.
                Defaults to "xavier".
        """
        super().__init__()
        # Setting up Encoder block
        self.encoder_channels = [in_channels, *encoder_channels]
        self.encoder = Encoder(self.encoder_channels)
        # Setting up Decoder block
        self.decoder_features = decoder_features
        self.decoder = Decoder(self.decoder_features, num_labels)
        # Flatten layer
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Makes a forward pass.

        Args:
            x (torch.Tensor): Input tensor for a CNN.

        Returns:
            torch.Tensor: Output tensor of a CNN.
        """
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.decoder(x)

        return x


class BasicBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1
    ) -> None:
        """_summary_

        Args:
            in_channels (int): _description_
            out_channels (int): _description_
            stride (int, optional): _description_. Defaults to 1.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(
        self, block: BasicBlock, num_blocks: list[int], num_classes: int = 10
    ) -> None:
        """_summary_

        Args:
            block (BasicBlock): _description_
            num_blocks (List[int]): _description_
            num_classes (int, optional): _description_. Defaults to 10.
        """
        super().__init__()
        self.in_channels = 16

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(
        self,
        block: BasicBlock,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """_summary_

        Args:
            block (BasicBlock): _description_
            out_channels (int): _description_
            num_blocks (int): _description_
            stride (int): _description_

        Returns:
            nn.Sequential: _description_
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def ednet4() -> CNN:
    """_summary_

    Returns:
        CNN: _description_
    """
    return CNN(
        in_channels=3,
        encoder_channels=(32, 64),
        decoder_features=(576, 250),
        num_labels=10,
    )


def resnet20() -> ResNet:
    """_summary_

    Returns:
        ResNet: _description_
    """
    return ResNet(BasicBlock, [3, 3, 3])
