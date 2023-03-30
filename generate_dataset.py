import torch
import torchaudio
import os

from config import SLICE_LEN, STRIDE, raw_clean_dir, raw_noisy_dir, clean_dir, noisy_dir
from dataset import create_chunks


def split_save(raw_dir, wav_name, target_dir):
    path = f"{raw_dir}/{wav_name}"
    clean, sr = torchaudio.load(path)
    if sr != 16000:
        clean = torchaudio.functional.resample(clean, sr, 16000)
    chunks = create_chunks(clean).squeeze()
    assert len(chunks.shape) == 2
    for i, chunk in enumerate(chunks):
        torchaudio.save(f"{target_dir}/{wav_name}_{i}.wav", chunk.unsqueeze(0), 16000)


train_files = []
with open('train_files.txt', 'r') as f:
    for line in f:
        train_files.append(line.strip())
clean_train_dir = f"{clean_dir}/train"
os.makedirs(clean_train_dir, exist_ok=True)
for wav_name in train_files:
    split_save(raw_clean_dir, wav_name, clean_train_dir)

noisy_train_dir = f"{noisy_dir}/train"
os.makedirs(noisy_train_dir, exist_ok=True)
for wav_name in train_files:
    split_save(raw_noisy_dir, wav_name, noisy_train_dir)

val_files = []
with open('val_files.txt', 'r') as f:
    for line in f:
        val_files.append(line.strip())
clean_val_dir = f"{clean_dir}/val"
os.makedirs(clean_val_dir, exist_ok=True)
for wav_name in val_files:
    split_save(raw_clean_dir, wav_name, clean_val_dir)

noisy_val_dir = f"{noisy_dir}/val"
os.makedirs(noisy_val_dir, exist_ok=True)
for wav_name in val_files:
    split_save(raw_noisy_dir, wav_name, noisy_val_dir)
