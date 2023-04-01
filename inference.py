import glob

import numpy as np
import torch
import torchaudio
from scipy.io.wavfile import write as wavwrite

from config import decoder_channels, encoder_channels, kernel_size, padding, stride
from model import CRN

if __name__ == "__main__":
    model = CRN(encoder_channels, decoder_channels, kernel_size, stride, padding)

    checkpoint = torch.load("model.pth", map_location="cpu")
    model.load_state_dict(checkpoint)

    clean_path = "datasets/clean_testset_wav"
    noisy_path = "datasets/noisy_testset_wav"
    # generate noisy file
    clean_files = glob.glob("datasets/clean_testset_wav/**/*.wav", recursive=True)
    clean_idx = np.random.randint(0, len(clean_files))
    filename = clean_files[clean_idx].split("/")[-1]

    clean, sr = torchaudio.load(f"{clean_path}/{filename}")
    if sr != 16000:
        clean = torchaudio.functional.resample(clean, sr, 16000)
    noisy, sr = torchaudio.load(f"{noisy_path}/{filename}")
    if sr != 16000:
        noisy = torchaudio.functional.resample(noisy, sr, 16000)

    wavwrite("clean.wav", 16000, clean.squeeze().numpy().astype(np.float32))
    wavwrite("noisy.wav", 16000, noisy.squeeze().numpy().astype(np.float32))
    estimated = model.inference("noisy.wav", device="cpu")
    wavwrite("estimated.wav", 16000, estimated.squeeze().numpy().astype(np.float32))
