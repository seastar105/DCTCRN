# dataset directories
raw_clean_dir = "datasets/clean_trainset_28spk_wav"
raw_noisy_dir = "datasets/noisy_trainset_28spk_wav"
clean_dir = "datasets/clean"
noisy_dir = "datasets/noisy"
rir_dir = "datasets/rir"
noise_dir = "datasets/noise"

# dataset parameters
snr_range = (-10, 20 + 1)
db_range = (-35, -15 + 1)
target_sr = 16000

SLICE_LEN = 16384  # 1 second
STRIDE = SLICE_LEN // 2  # 50% overlap
FRAME_LEN = 512
HOP_LEN = 128

SEED = 998244353
MAX_BATCH_SIZE = 64
EPOCHS = 300
LEARNING_RATE = 0.001
encoder_channels = [1, 8, 16, 32, 64, 128, 128, 256]
decoder_channels = [256, 128, 128, 64, 32, 16, 8, 1]
kernel_size = (2, 5)
stride = (1, 2)
padding = (0, 0)

PROJECT_NAME = "dae-dctcrn"
RUN_NAME = "test-run"
CHECKPOINT_PATH = "./checkpoints.tar"
