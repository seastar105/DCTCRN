from typing import Optional, Tuple

import numpy as np
import torch
import torch_dct
from torchaudio.functional import fftconvolve

EPS = np.finfo(np.float32).eps


def normalize(audio: torch.Tensor, target_level: int = -25) -> torch.Tensor:
    """Normalize the signal to the target level"""
    rms = (audio**2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    audio = audio * scalar
    return audio


def make_reverb(clean: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
    reverb_speech = fftconvolve(clean, rir, mode="full")
    return reverb_speech[:, : clean.shape[1]]


def make_noisy(
    clean: torch.Tensor,
    rir: torch.Tensor,
    noise: torch.Tensor,
    snr_range: Tuple[int, int],
    db_range: Tuple[int, int],
    target_level: int = -25,
) -> torch.Tensor:
    # reference:
    # https://github.com/microsoft/DNS-Challenge/blob/a2c7487e12d06d709aeebe5659c21bbf6e1a47aa/audiolib.py#L155
    snr = np.random.randint(snr_range[0], snr_range[1])
    if clean.shape[1] > noise.shape[1]:
        noise = torch.nn.functional.pad(noise, (0, clean.shape[1] - noise.shape[1]))
    else:
        noise = noise[:, : clean.shape[1]]
    # normalize
    reverb = make_reverb(clean, rir)
    reverb /= reverb.abs().max() + EPS
    reverb = normalize(reverb, target_level)
    rms_reverb = (reverb**2).mean() ** 0.5

    clean /= clean.abs().max() + EPS
    clean = normalize(clean, target_level)

    noise /= noise.abs().max() + EPS
    noise = normalize(noise, target_level)
    rms_noise = (noise**2).mean() ** 0.5

    # Set the noise level for a given SNR
    noise_scaler = rms_reverb / (rms_noise + EPS) / (10 ** (snr / 20.0))
    noise = noise * noise_scaler

    noisy = reverb + noise

    noisy_target_level = np.random.randint(db_range[0], db_range[1])
    rms_noisy = (noisy**2).mean() ** 0.5
    noisy_scaler = 10 ** (noisy_target_level / 20) / (rms_noisy + EPS)
    clean = clean * noisy_scaler
    noisy = noisy * noisy_scaler
    noise = noise * noisy_scaler

    return clean.squeeze(), noisy.squeeze(), noise.squeeze()


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


def synthesize_wav(
    dct: torch.Tensor, frame_length: int, hop_length: int, window: Optional[torch.Tensor] = None
) -> torch.Tensor:
    framed = dct.permute(0, 2, 1)
    istdct = torch_dct.idct(framed, norm="ortho")
    # overlap and add
    signal = torch.zeros(istdct.shape[0], istdct.shape[1] * hop_length + frame_length - hop_length)
    overlap = torch.zeros(istdct.shape[0], istdct.shape[1] * hop_length + frame_length - hop_length)
    for i in range(istdct.shape[1]):
        signal[:, i * hop_length : i * hop_length + frame_length] += istdct[:, i, :]
    # I'm not sure if this is correct, but reconstructed wave form wat too loud not applying this.
    print(f"signal.max = {signal.max()}")
    print(f"overlap.max = {overlap.max()}")
    return signal
