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


dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)  # , collate_fn=collate_fn)
frame_len = 512
hop_len = 128
window = torch.hamming_window(frame_len)

for clean, noisy in dataloader:
    print(clean.shape, noisy.shape)
    print(f"clean.max() = {clean.max()}")
    clean_dct = stdct(clean, frame_len, hop_len, window=window)
    noisy_dct = stdct(noisy, frame_len, hop_len, window=window)
    icm = clean_dct / noisy_dct
    print(f"clean_dct.shape: {clean_dct.shape}")
    print(f"noisy_dct.shape: {noisy_dct.shape}")

    clean_wav = synthesize_wav(clean_dct, frame_len, hop_len, window=window)
    estimate_wav = synthesize_wav(icm * noisy_dct, frame_len, hop_len, window=window)
    print(f"clean_wav.shape: {clean_wav.shape}")
    print(f"estimate_wav.shape: {estimate_wav.shape}")

    assert clean_wav.shape == estimate_wav.shape
    # for i in range(clean_wav.shape[0]):
    #     wavwrite(f"clean_{i}.wav", 16000, clean_wav[i].squeeze().numpy().astype(np.float32))
    #     wavwrite(f"noisy_{i}.wav", 16000, noisy[i].squeeze().numpy().astype(np.float32))
    #     wavwrite(f"estimate_{i}.wav", 16000, estimate_wav[i].squeeze().numpy().astype(np.float32))
    wavwrite("clean.wav", 16000, clean_wav.squeeze().numpy().astype(np.float32))
    wavwrite("noisy.wav", 16000, noisy.squeeze().numpy().astype(np.float32))
    wavwrite("estimate.wav", 16000, estimate_wav.squeeze().numpy().astype(np.float32))
    break
