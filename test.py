import glob

import numpy as np
import torchaudio
from scipy.io.wavfile import write as wavwrite
from torchaudio.functional import fftconvolve, resample

from utils import make_noisy

clean_files = glob.glob("datasets/clean/**/*.wav", recursive=True)
rir_files = glob.glob(
    "datasets/impulse_responses/SLR28/RIRS_NOISES/real_rirs_isotropic_noises/**/*.wav", recursive=True
)
noise_files = glob.glob("datasets/noise/**/*.wav", recursive=True)

clean_idx = np.random.randint(0, len(clean_files))
rir_idx = np.random.randint(0, len(rir_files))
noise_idx = np.random.randint(0, len(noise_files))

clean_speech, sr = torchaudio.load(clean_files[clean_idx])
print(f"clean_speech.shape: {clean_speech.shape} {sr}")
clean_speech = resample(clean_speech, sr, 16000)
rir_speech, sr = torchaudio.load(rir_files[rir_idx])
print(f"rir_speech.shape: {rir_speech.shape} {sr}")
noise, sr = torchaudio.load(noise_files[noise_idx])
print(f"noise.shape: {noise.shape} {sr}")

rir_channel = np.random.randint(0, rir_speech.shape[0])
rir_speech = rir_speech[rir_channel].unsqueeze(0)
reverb_speech = fftconvolve(clean_speech, rir_speech, mode="full")
reverb_speech = reverb_speech[:, : clean_speech.shape[1]]
print(f"reverb_speech.shape: {reverb_speech.shape}")

result = make_noisy(clean_speech, noise, (-10, -9))

wavwrite("clean.wav", 16000, clean_speech.squeeze().numpy().astype(np.float32))
wavwrite("result.wav", 16000, result.squeeze().numpy().astype(np.float32))
