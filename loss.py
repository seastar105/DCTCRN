import torch


def si_snr(clean, estimated):
    eps = torch.finfo(estimated.dtype).eps

    clean = clean - torch.mean(clean, dim=-1, keepdim=True)
    estimated = estimated - torch.mean(estimated, dim=-1, keepdim=True)

    alpha = (torch.sum(estimated * clean, dim=-1, keepdim=True) + eps) / (
        torch.sum(clean**2, dim=-1, keepdim=True) + eps
    )
    target_scaled = alpha * clean

    noise = target_scaled - estimated

    val = (torch.sum(target_scaled**2, dim=-1) + eps) / (torch.sum(noise**2, dim=-1) + eps)
    val = 10 * torch.log10(val)
    return val.sum() / val.numel()
