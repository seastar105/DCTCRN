from typing import List, Tuple

import torch
import torch.nn as nn
import torchaudio

from config import SLICE_LEN, STRIDE
from dataset import create_chunks
from utils import istdct, stdct


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
        self.lstm.flatten_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)
        return x


class CRN(nn.Module):
    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        kernel_size: Tuple[int, int] = (2, 5),
        stride: Tuple[int, int] = (1, 2),
        padding: Tuple[int, int] = (1, 0),
        hidden_dim: int = 256,
    ):
        super(CRN, self).__init__()
        self.encoder = Encoder(encoder_channels, kernel_size, stride, padding)
        self.processor = Processor(hidden_dim)
        self.decoder = Decoder(decoder_channels, kernel_size, stride, padding)

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encs = self.encoder(x)
        x = encs[-1]
        T = x.shape[-2]
        x = x.reshape(x.shape[0], T, -1)
        x = self.processor(x)
        x = x.reshape(x.shape[0], encs[-1].shape[1], T, -1)
        x = self.decoder(x, encs)
        return x

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight.data, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias.data)
        elif isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(module.weight.data, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias.data)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight.data)
            if module.bias is not None:
                nn.init.zeros_(module.bias.data)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, mean=0.0, std=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias.data)

    @torch.inference_mode()
    def inference(self, file_path: str, device="cuda", normalize=False) -> torch.Tensor:
        audio, sr = torchaudio.load(file_path)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        noisy = create_chunks(audio)
        from config import FRAME_LEN, HOP_LEN

        noisy = noisy.to(device)
        window = torch.hann_window(FRAME_LEN).to(noisy.device)
        noisy_dct = stdct(noisy, FRAME_LEN, HOP_LEN, window=window)
        mask = torch.nn.functional.tanh(self.forward(noisy_dct))
        recon = istdct(noisy_dct * mask, FRAME_LEN, HOP_LEN, window=window)
        recon = recon.to("cpu")
        window = torch.ones(recon.shape[-1])
        clean = torch.zeros_like(audio)
        overlap = torch.zeros_like(audio)
        for i in range(recon.shape[0] - 1):
            clean[:, i * STRIDE : i * STRIDE + SLICE_LEN] += recon[i]
            overlap[:, i * STRIDE : i * STRIDE + SLICE_LEN] += window
        if (recon.shape[0] - 1) * STRIDE + SLICE_LEN < clean.shape[-1]:
            clean[:, (recon.shape[0] - 1) * STRIDE : (recon.shape[0] - 1) * STRIDE + SLICE_LEN] += recon[-1]
            overlap[:, (recon.shape[0] - 1) * STRIDE : (recon.shape[0] - 1) * STRIDE + SLICE_LEN] += window
        else:
            clean[:, -SLICE_LEN:] += recon[-1]
            overlap[:, -SLICE_LEN:] += window
        clean = clean / overlap
        if normalize:
            clean = clean / torch.max(torch.abs(clean))
        return clean
