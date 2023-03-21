import numpy as np
import torch
from scipy.io.wavfile import write as wavwrite
from torch.utils.data import DataLoader

from config import clean_dir, noisy_dir, target_sr
from dataset import CleanNoisyDataset
from utils import stdct, synthesize_wav

dataset = CleanNoisyDataset(clean_dir=clean_dir, noisy_dir=noisy_dir, target_sr=target_sr)


dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
frame_len = 512
hop_len = 128
window = torch.hamming_window(frame_len)

for clean, noisy in dataloader:
    clean_dct = stdct(clean, frame_len, hop_len, window=window)
    noisy_dct = stdct(noisy, frame_len, hop_len, window=window)
    clean_stft = torch.stft(clean, n_fft=frame_len, hop_length=hop_len, return_complex=True)
    icm = clean_dct / noisy_dct

    clean_wav = synthesize_wav(clean_dct, frame_len, hop_len, window=window)
    stft_wav = torch.istft(clean_stft, n_fft=frame_len, hop_length=hop_len)
    noisy_wav = synthesize_wav(noisy_dct, frame_len, hop_len, window=window)
    estimate_wav = synthesize_wav(icm * noisy_dct, frame_len, hop_len, window=window)

    wavwrite("clean.wav", 16000, clean_wav.squeeze().numpy().astype(np.float32))
    wavwrite("stft.wav", 16000, stft_wav.squeeze().numpy().astype(np.float32))
    wavwrite("noisy.wav", 16000, noisy_wav.squeeze().numpy().astype(np.float32))
    wavwrite("estimate.wav", 16000, estimate_wav.squeeze().numpy().astype(np.float32))
    break
