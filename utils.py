import numpy as np
import torch
from torchaudio.functional import add_noise, fftconvolve


def make_reverb(clean: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
    reverb_speech = fftconvolve(clean, rir, mode="full")
    return reverb_speech[:, : clean.shape[1]]


def make_noisy(clean: torch.Tensor, noise: torch.Tensor, snr_range: tuple) -> torch.Tensor:
    snr = np.random.randint(snr_range[0], snr_range[1])
    if clean.shape[1] > noise.shape[1]:
        # make noise repeat
        noise = noise.repeat(1, clean.shape[1] // noise.shape[1] + 1)
    else:
        noise = noise[:, : clean.shape[1]]
    noisy = add_noise(clean, noise, torch.IntTensor([snr]))
    return noisy
