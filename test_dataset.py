import numpy as np
import torch
from scipy.io.wavfile import write as wavwrite
from torch.utils.data import DataLoader

from config import clean_dir, db_range, noise_dir, rir_dir, snr_range, target_sr
from dataset import CleanNoisyDataset
from utils import stdct, synthesize_wav

dataset = CleanNoisyDataset(
    clean_dir=clean_dir,
    rir_dir=rir_dir,
    noisy_dir=noise_dir,
    target_sr=target_sr,
    snr_range=snr_range,
    db_range=db_range,
)


dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
frame_len = 512
hop_len = 128
window = torch.hamming_window(frame_len)

for clean, noisy, noise in dataloader:
    clean_dct = stdct(clean, frame_len, hop_len, window=window)
    noise_dct = stdct(noise, frame_len, hop_len, window=window)
    noisy_dct = stdct(noisy, frame_len, hop_len, window=window)
    icm = clean_dct / noisy_dct

    clean_wav = synthesize_wav(clean_dct, frame_len, hop_len, window=window)
    estimate_wav = synthesize_wav(icm * noisy_dct, frame_len, hop_len, window=window)

    wavwrite("clean.wav", 16000, clean_wav.squeeze().numpy().astype(np.float32))
    wavwrite("noisy.wav", 16000, noisy.squeeze().numpy().astype(np.float32))
    wavwrite("estimate.wav", 16000, estimate_wav.squeeze().numpy().astype(np.float32))
