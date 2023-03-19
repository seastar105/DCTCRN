from typing import Optional

import numpy as np
import torch
import torch_dct
from torchaudio.functional import add_noise, fftconvolve

EPS = np.finfo(np.float32).eps


def make_reverb(clean: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
    reverb_speech = fftconvolve(clean, rir, mode="full")
    return reverb_speech[:, : clean.shape[1]]


def make_noisy(clean: torch.Tensor, noise: torch.Tensor, snr_range: tuple) -> torch.Tensor:
    snr = np.random.randint(snr_range[0], snr_range[1])
    if clean.shape[1] > noise.shape[1]:
        noise = torch.nn.functional.pad(noise, (0, clean.shape[1] - noise.shape[1]))
    else:
        noise = noise[:, : clean.shape[1]]
    # normalize
    clean /= clean.abs().max() + EPS
    noise /= noise.abs().max() + EPS
    noisy = add_noise(clean, noise, torch.IntTensor([snr]))
    return noisy


def frame(
    signal: torch.Tensor, frame_length: int, hop_length: int, window: Optional[torch.Tensor] = None
) -> torch.Tensor:
    framed = signal.unfold(-1, frame_length, hop_length)
    if window is not None:
        framed *= window
    return framed


def stdct(
    signal: torch.Tensor, frame_length: int, hop_length: int, window: Optional[torch.Tensor] = None
) -> torch.Tensor:
    framed = frame(signal, frame_length, hop_length, window=window)
    dct = torch_dct.dct(framed, norm="ortho")
    return dct.permute(0, 2, 1)


def synthesize_wav(dct: torch.Tensor, frame_length: int, hop_length: int) -> torch.Tensor:
    framed = dct.permute(0, 2, 1)
    istdct = torch_dct.idct(framed, norm="ortho")
    # overlap and add
    signal = torch.zeros(istdct.shape[0], istdct.shape[1] * hop_length + frame_length - hop_length)
    window = torch.hamming_window(frame_length)
    istdct *= window
    for i in range(istdct.shape[1]):
        signal[:, i * hop_length : i * hop_length + frame_length] += istdct[:, i, :]
    # I'm not sure if this is correct, but reconstructed wave form wat too loud not applying this.
    return signal / signal.abs().max()
