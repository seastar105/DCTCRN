import torch
from torch.utils.data import DataLoader

from config import clean_dir, noisy_dir, target_sr
from dataset import CleanNoisyDataset, collate_fn
from utils import istdct, stdct


def si_snr(clean, estimated):
    eps = torch.finfo(estimated.dtype).eps
    clean = clean.squeeze()
    estimated = estimated.squeeze()

    # from https://github.com/Lightning-AI/metrics/blob/v0.11.4/src/torchmetrics/functional/audio/sdr.py#L206
    clean = clean - torch.mean(clean, dim=-1, keepdim=True)
    estimated = estimated - torch.mean(estimated, dim=-1, keepdim=True)

    alpha = (torch.sum(estimated * clean, dim=-1, keepdim=True) + eps) / (
        torch.sum(clean**2, dim=-1, keepdim=True) + eps
    )
    target_scaled = alpha * clean

    noise = target_scaled - clean

    val = (torch.sum(target_scaled**2, dim=-1) + eps) / (torch.sum(noise**2, dim=-1) + eps)
    val = 10 * torch.log10(val)
    return val


def si_snr_loss(clean, estimated):
    return -si_snr(clean, estimated)


if __name__ == "__main__":
    dataset = CleanNoisyDataset(clean_dir=clean_dir, noisy_dir=noisy_dir, target_sr=target_sr)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
    for clean, noisy in dataloader:
        window = torch.hamming_window(512)
        clean_dct = stdct(clean, 512, 128, window=window)
        noisy_dct = stdct(noisy, 512, 128, window=window)

        mask = clean_dct / noisy_dct
        estimate = mask * noisy_dct
        estimate = istdct(estimate, 512, 128, window=window)
        print(si_snr_loss(clean, clean))
        print(si_snr_loss(clean, estimate))
        print(si_snr_loss(clean, torch.zeros_like(clean)))
        break
