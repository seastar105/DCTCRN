# import glob
#
# import numpy as np
# import torchaudio
# from scipy.io.wavfile import write as wavwrite
# from torchaudio.functional import fftconvolve, resample
#
# from utils import make_noisy
#
# clean_files = glob.glob("datasets/clean/**/*.wav", recursive=True)
# rir_files = glob.glob("datasets/rir/**/*.wav", recursive=True)
# noise_files = glob.glob("datasets/noise/**/*.wav", recursive=True)
#
# clean_idx = np.random.randint(0, len(clean_files))
# rir_idx = np.random.randint(0, len(rir_files))
# noise_idx = np.random.randint(0, len(noise_files))
#
# clean_speech, sr = torchaudio.load(clean_files[clean_idx])
# print(f"clean_speech.shape: {clean_speech.shape} {sr}")
# clean_speech = resample(clean_speech, sr, 16000)
# rir_speech, sr = torchaudio.load(rir_files[rir_idx])
# print(f"rir_speech.shape: {rir_speech.shape} {sr}")
# noise, sr = torchaudio.load(noise_files[noise_idx])
# print(f"noise.shape: {noise.shape} {sr}")
#
# rir_speech = rir_speech.mean(0, keepdim=True)
# # rir_speech = rir_speech[0].unsqueeze(0)
# reverb_speech = fftconvolve(clean_speech, rir_speech, mode="full")
# reverb_speech = reverb_speech[:, : clean_speech.shape[1]]
# print(f"reverb_speech.shape: {reverb_speech.shape}")
#
# clean, noisy = make_noisy(clean_speech, noise, (-10, 21), (-35, -14))
# # result = reverb_speech
#
# wavwrite("clean.wav", 16000, clean.squeeze().numpy().astype(np.float32))
# wavwrite("noisy.wav", 16000, noisy.squeeze().numpy().astype(np.float32))

import numpy as np
import torch
import torchaudio
from scipy.io.wavfile import write as wavwrite

from utils import stdct, synthesize_wav

frame_length = 512
hop_length = 128

y, sr = torchaudio.load("datasets/clean_trainset_28spk_wav/p278_195.wav")
if sr != 16000:
    y = torchaudio.functional.resample(y, sr, 16000)
print(y.shape)

window = torch.hann_window(frame_length)
stft = torch.stft(y, n_fft=frame_length, hop_length=hop_length, onesided=False, return_complex=True)
print(stft.shape)
dct = stdct(y, frame_length, hop_length, window=torch.hann_window(frame_length))
print(dct.shape)

stft_recon = torch.istft(stft, n_fft=frame_length, hop_length=hop_length, onesided=False)
print(stft_recon.shape)
wavwrite("stft_recon.wav", 16000, stft_recon.squeeze().numpy().astype(np.float32))
stdct_recon = synthesize_wav(dct, frame_length, hop_length, window=torch.hann_window(frame_length))
print(stdct_recon.shape)
wavwrite("stdct_recon.wav", 16000, stdct_recon.squeeze().numpy().astype(np.float32))
