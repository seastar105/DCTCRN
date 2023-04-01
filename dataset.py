import glob
from typing import Tuple

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from config import MAX_BATCH_SIZE, SLICE_LEN, STRIDE
from utils import frame


class CleanNoisyDataset(Dataset):
    def __init__(self, clean_dir: str, noisy_dir: str, target_sr: int = 16000):
        super(CleanNoisyDataset, self).__init__()
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.target_sr = target_sr

        self.files = []

        for file in glob.glob(self.clean_dir + "/**/*.wav", recursive=True):
            self.files.append(file.split("/")[-1])
        np.random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        filename = self.files[index]
        clean_file = self.clean_dir + "/" + filename
        noisy_file = self.noisy_dir + "/" + filename

        clean, sr = torchaudio.load(clean_file)
        if sr != self.target_sr:
            clean = torchaudio.functional.resample(clean, sr, self.target_sr)
        noisy, sr = torchaudio.load(noisy_file)
        if sr != self.target_sr:
            noisy = torchaudio.functional.resample(noisy, sr, self.target_sr)
        assert clean.shape[0] == 1, f"clean speech should be mono {clean.shape}"
        assert clean.shape == noisy.shape, f"clean and noisy should have same shape {clean.shape} {noisy.shape}"
        return clean, noisy


def create_chunks(wav):
    # wav: (1, L)
    # chunks: (N, 1, SLICE_LEN)
    chunks = frame(wav, frame_length=SLICE_LEN, hop_length=STRIDE).permute(1, 0, 2)
    if wav.shape[-1] > (chunks.shape[0] - 1) * STRIDE + SLICE_LEN:
        last_chunk = wav[:, -SLICE_LEN:].unsqueeze(1)
        chunks = torch.vstack([chunks, last_chunk])
    return chunks


def collate_fn(batch):
    clean_wavs = []
    noisy_wavs = []
    for clean, noisy in batch:
        clean = create_chunks(clean)
        noisy = create_chunks(noisy)
        clean_wavs.append(clean)
        noisy_wavs.append(noisy)
    clean, noisy = torch.vstack(clean_wavs), torch.vstack(noisy_wavs)
    if clean.shape[0] > MAX_BATCH_SIZE:
        clean = clean[:MAX_BATCH_SIZE]
        noisy = noisy[:MAX_BATCH_SIZE]
    return clean, noisy
