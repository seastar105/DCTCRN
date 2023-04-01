import torch
import torchaudio

from loss import si_snr
from model import Decoder, Encoder, Processor
from utils import frame, istdct, stdct

clean, sr = torchaudio.load("clean.wav")
if sr != 16000:
    clean = torchaudio.functional.resample(clean, sr, 16000)
noisy, sr = torchaudio.load("noisy.wav")
if sr != 16000:
    noisy = torchaudio.functional.resample(noisy, sr, 16000)

frame_length = 512
hop_length = 128
window = torch.hann_window(frame_length)
slice_length = 16384
slice_stride = slice_length // 2
kernel_size = (2, 5)
stride = (1, 2)
padding = (0, 0)


encoder_channels = [1, 8, 16, 32, 64, 128, 128, 256]
decoder_channels = [256, 128, 128, 64, 32, 16, 8, 1]
hidden_dim = 256

from model import CRN
model = CRN(encoder_channels, decoder_channels, kernel_size, stride, padding, hidden_dim)

audio = model.inference('noisy.wav', device='cpu')
print(audio.shape)
import soundfile as sf
import numpy as np
sf.write('tmp.wav', audio.cpu().squeeze().numpy().astype(np.float32), 16000)