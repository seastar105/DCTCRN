# dataset directories
clean_dir = "datasets/clean_trainset_28spk_wav"
noisy_dir = "datasets/noisy_trainset_28spk_wav"
rir_dir = "datasets/rir"
noise_dir = "datasets/noise"

# dataset parameters
snr_range = (-10, 20 + 1)
db_range = (-35, -15 + 1)
target_sr = 16384

SLICE_LEN = 16000  # 1 second
STRIDE = SLICE_LEN // 2  # 50% overlap
