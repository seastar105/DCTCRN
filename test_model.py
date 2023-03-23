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

clean_batch = frame(clean, slice_length, slice_stride).permute(1, 0, 2)
noisy_batch = frame(noisy, slice_length, slice_stride).permute(1, 0, 2)

print(noisy_batch.shape)  # (B, C, T)

clean_dct = stdct(clean_batch, frame_length, hop_length, window=window)
noisy_dct = stdct(noisy_batch, frame_length, hop_length, window=window)

print(noisy_dct.shape)  # (B, C, T, F)

encoder_channels = [1, 8, 16, 32, 64, 128, 128, 256]
encoder = Encoder(encoder_channels, kernel_size, stride, padding)

outs = encoder(noisy_dct)
for out in outs:
    print(out.shape)

last_output = outs[-1]
last_output = last_output.reshape(last_output.shape[0], last_output.shape[-2], -1)  # (B, C, T, F) -> (B, T, C*F)

print(last_output.shape)

hidden_dim = 256
processor = Processor(hidden_dim)
out = processor(last_output)

print(out.shape)

out = out.reshape(out.shape[0], outs[-1].shape[1], out.shape[-2], -1)  # (B, T, C*F) -> (B, C, T, F)
print(out.shape)

decoder_channels = [256, 128, 128, 64, 32, 16, 8, 1]
decoder = Decoder(decoder_channels, kernel_size, stride, padding)
out = decoder(out, outs)
print(out.shape)

estimated = istdct(out * noisy_dct, frame_length, hop_length, window=window)

loss = si_snr(clean_batch.squeeze(), estimated.squeeze())

loss.backward()
