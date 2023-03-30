import time

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb
from config import (
    CHECKPOINT_PATH,
    EPOCHS,
    BATCH_SIZE,
    FRAME_LEN,
    HOP_LEN,
    LEARNING_RATE,
    SEED,
    clean_dir,
    decoder_channels,
    encoder_channels,
    kernel_size,
    noisy_dir,
    padding,
    stride,
    target_sr,
)
from dataset import CleanNoisyDataset, collate_fn
from loss import si_snr_loss
from model import CRN
from utils import istdct, stdct
import os


def predefines():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False

    # seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def train_loop(model, loader, optimizer, global_step, device="cuda"):
    model.train()
    window = torch.hann_window(FRAME_LEN).to(device)
    start = time.time()
    losses = []
    for clean, noisy in loader:
        clean = clean.to(device)
        noisy = noisy.to(device)
        noisy_dct = stdct(noisy, FRAME_LEN, HOP_LEN, window=window)
        mask = torch.nn.functional.tanh(model(noisy_dct))
        estimated = istdct(noisy_dct * mask, FRAME_LEN, HOP_LEN, window=window)
        loss = si_snr_loss(clean, estimated)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        losses.append(loss.item())
        if global_step % 50 == 0:
            elapsed = time.time() - start
            start = time.time()
            train_loss = sum(losses) / len(losses)
            print(f"Step: {global_step}, Loss: {train_loss}, Time: {elapsed} sec")
            wandb.log(
                {"train/loss": train_loss, "train/step_time": elapsed, "train/steps": global_step},
            )
            losses = []
    return global_step


def val_loop(model, loader, infer_file, epoch, device="cuda"):
    model.eval()
    window = torch.hann_window(FRAME_LEN).to(device)
    with torch.inference_mode():
        losses = []
        for clean, noisy in loader:
            clean = clean.to(device)
            noisy = noisy.to(device)
            noisy_dct = stdct(noisy, FRAME_LEN, HOP_LEN, window=window)
            mask = torch.nn.functional.tanh(model(noisy_dct))
            estimated = istdct(noisy_dct * mask, FRAME_LEN, HOP_LEN, window=window)
            loss = si_snr_loss(clean, estimated)
            losses.append(loss.item())
        val_loss = sum(losses) / len(losses)
        audio = model.inference(infer_file)
        log_audio = wandb.Audio(
            audio.cpu().squeeze().numpy().astype(np.float32), sample_rate=16000, caption="Inference"
        )
        print(f"Val Loss: {val_loss}")
        wandb.log({"val/loss": val_loss, "val/sample": log_audio, "epoch": epoch})
    return val_loss


if __name__ == "__main__":
    predefines()
    wandb.init(project="dctcrn", name="dctcrn-t-voicebank")
    wandb.config.update(
        {"bacth_size": BATCH_SIZE, "epochs": EPOCHS, "dataset": "voicebank", "seed": SEED}, allow_val_change=True
    )
    if not torch.cuda.is_available():
        assert False, "CUDA is not available"
    device = "cuda"
    model = CRN(encoder_channels, decoder_channels, kernel_size, stride, padding)
    # print the number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    global_step = 0
    epoch = 1
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)
    if wandb.run.resumed:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    train_clean_dir = os.path.join(clean_dir, "train")
    train_noisy_dir = os.path.join(noisy_dir, "train")
    train_ds = CleanNoisyDataset(train_clean_dir, train_noisy_dir, target_sr)
    val_clean_dir = os.path.join(clean_dir, "val")
    val_noisy_dir = os.path.join(noisy_dir, "val")
    val_ds = CleanNoisyDataset(val_clean_dir, val_noisy_dir, target_sr)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
    )

    model = model.to(device)
    model = torch.compile(model)
    for epoch in range(epoch, EPOCHS+1):
        epoch_start = time.time()
        global_step = train_loop(model, train_loader, optimizer, global_step, device)
        epoch_end = time.time()
        print(f"Epoch: {epoch}, Time: {epoch_end - epoch_start} sec")
        wandb.log({"train/epoch_time": epoch_end - epoch_start, "epoch": epoch})
        epoch_start = time.time()
        val_loss = val_loop(model, val_loader, "noisy.wav", epoch, device)
        epoch_end = time.time()
        print(f"Validation: {epoch}, Loss: {val_loss} Time: {epoch_end - epoch_start} sec")
        wandb.log({"val/epoch_time": epoch_end - epoch_start, "epoch": epoch})
        if epoch >= 100:
            scheduler.step(val_loss)
        # save model every epoch
        torch.save(model.state_dict(), f"checkpoints/{epoch}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "global_step": global_step,
            },
            CHECKPOINT_PATH,
        )
        wandb.save(CHECKPOINT_PATH)

    wandb.finish()
