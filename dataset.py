import glob
from typing import Tuple

import numpy as np
import torchaudio
from torch.utils.data import Dataset

from utils import make_noisy


class CleanNoisyDataset(Dataset):
    def __init__(
        self,
        clean_dir: str,
        rir_dir: str,
        noisy_dir: str,
        target_sr: int = 16000,
        snr_range: Tuple[int, int] = (-10, 20 + 1),
        db_range: Tuple[int, int] = (-35, -15 + 1),
    ):
        self.clean_dir = clean_dir
        self.rir_dir = rir_dir
        self.noisy_dir = noisy_dir

        self.clean_files = glob.glob(self.clean_dir + "/**/*.wav", recursive=True)
        self.rir_files = glob.glob(self.rir_dir + "/**/*.wav", recursive=True)
        self.noisy_files = glob.glob(self.noisy_dir + "/**/*.wav", recursive=True)

        self.target_sr = target_sr
        self.snr_range = snr_range
        self.db_range = db_range

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, index):
        clean_file = self.clean_files[index]
        rir_idx = np.random.randint(0, len(self.rir_files))

        clean, sr = torchaudio.load(clean_file)
        if sr != self.target_sr:
            clean = torchaudio.functional.resample(clean, sr, self.target_sr)
        rir, sr = torchaudio.load(self.rir_files[rir_idx])
        if sr != self.target_sr:
            rir = torchaudio.functional.resample(rir, sr, self.target_sr)
        noise, sr = torchaudio.load(self.noisy_files[index])
        if sr != self.target_sr:
            noise = torchaudio.functional.resample(noise, sr, self.target_sr)

        assert clean.shape[0] == 1, f"clean speech should be mono {clean.shape}"
        assert noise.shape[0] == 1, "noise should be mono"

        rir_channels = rir.shape[0]
        if rir_channels > 1:
            rir = rir.mean(0, keepdim=True)

        return make_noisy(clean, rir, noise, self.snr_range, self.db_range)
