from typing import List, Tuple

import torch
import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(EncoderLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self, channels: List[int], kernel_size: Tuple[int, int], stride: Tuple[int, int], padding: Tuple[int, int]
    ):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(channels[i], channels[i + 1], kernel_size, stride, padding) for i in range(len(channels) - 1)]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = []
        for layer in self.layers:
            x = torch.nn.functional.pad(x, (0, 0, 1, 0))
            x = layer(x)
            out.append(x)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, out_padding=(0, 0)):
        super(DecoderLayer, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False, output_padding=out_padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self, channels: List[int], kernel_size: Tuple[int, int], stride: Tuple[int, int], padding: Tuple[int, int]
    ):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(2 * channels[i], channels[i + 1], kernel_size, stride, padding)
                if i < len(channels) - 3
                else DecoderLayer(2 * channels[i], channels[i + 1], kernel_size, stride, padding, out_padding=(0, 1))
                for i in range(len(channels) - 1)
            ]
        )

    def forward(self, x: torch.Tensor, encs: List[torch.Tensor]) -> torch.Tensor:
        for layer, enc in zip(self.layers, encs[::-1]):
            inputs = [x, enc]
            x = torch.cat(inputs, dim=1)
            x = x[:, :, :-1, :]
            x = layer(x)
        return x


class Processor(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super(Processor, self).__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, 2, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)
        return x


class CRN(nn.Module):
    def __init__(
        self,
        channels: List[int],
        kernel_size: Tuple[int, int] = (2, 5),
        stride: Tuple[int, int] = (1, 2),
        padding: Tuple[int, int] = (1, 0),
        hidden_dim: int = 256,
    ):
        super(CRN, self).__init__()
        self.encoder = Encoder(channels, kernel_size, stride, padding)
        self.processor = Processor(hidden_dim)
        self.decoder = Decoder(channels[::-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encs = self.encoder(x)
        x = encs[-1].reshape(x.shape[0], x.shape[-2], -1)
        x = self.processor(x)
        x = self.decoder(x, encs)
        return x
