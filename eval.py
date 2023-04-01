import torch
import glob
import torchaudio
from pystoi import stoi
from pesq import pesq
from tqdm import tqdm


from config import (
    test_clean_dir,
    test_noisy_dir,
    target_sr,
    encoder_channels,
    decoder_channels,
    kernel_size,
    stride,
    padding
)
from model import CRN
from loss import si_snr


if __name__ == "__main__":
    model = CRN(encoder_channels, decoder_channels, kernel_size, stride, padding)
    checkpoint = torch.load('model.pth', map_location='cpu')
    model.load_state_dict(checkpoint)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.eval().to(device)

    test_clean_files = glob.glob(test_clean_dir + '**/*.wav', recursive=True)
    si_snr_scores = []
    stoi_scores = []
    pesq_scores = []
    for clean_file in tqdm(test_clean_files):
        filename = clean_file.split('/')[-1]
        noisy_file = test_noisy_dir + f'/{filename}'

        estimated = model.inference(noisy_file, device=device).cpu().squeeze()
        clean, sr = torchaudio.load(clean_file)
        if sr != target_sr:
            clean = torchaudio.functional.resample(clean, sr, target_sr)
        clean = clean.squeeze()
        si_snr_scores.append(si_snr(clean, estimated))
        clean = clean.numpy()
        estimated = estimated.numpy()
        stoi_scores.append(stoi(clean, estimated, target_sr, extended=False))
        pesq_scores.append(pesq(target_sr, clean, estimated, "wb"))
    print(f"SI-SNR: {sum(si_snr_scores) / len(si_snr_scores)}")
    print(f"STOI: {sum(stoi_scores) / len(stoi_scores)}")
    print(f"PESQ: {sum(pesq_scores) / len(pesq_scores)}")
