import glob

import numpy as np
import torch
import torchaudio

from config import decoder_channels, encoder_channels, kernel_size, padding, stride
from model import CRN

model = CRN(encoder_channels, decoder_channels, kernel_size, stride, padding)

checkpoint = torch.load("checkpoints_prev/120.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])

# generate noisy file
clean_files = glob.glob("datasets/clean/**/*.wav", recursive=True)
noises = glob.glob("datasets/noise/**/*.wav", recursive=True)

clean_idx = np.random.randint(0, len(clean_files))
clean, sr = torchaudio.load(clean_files[clean_idx])
if sr != 16000:
    clean = torchaudio.functional.resample(clean, sr, 16000)
noise_idx = np.random.randint(0, len(noises))
noise, sr = torchaudio.load(noises[noise_idx])
if sr != 16000:
    noise = torchaudio.functional.resample(noise, sr, 16000)

if clean.shape[-1] < noise.shape[-1]:
    noise = noise[:, : clean.shape[-1]]
else:
    clean = clean[:, : noise.shape[-1]]

noisy = torchaudio.functional.add_noise(clean, noise, torch.IntTensor([-2]))
torchaudio.save("clean.wav", clean, 16000)
torchaudio.save("noisy.wav", noisy, 16000)
estimated = model.inference("noisy.wav", device="cpu")
torchaudio.save("estimated.wav", estimated, 16000)
