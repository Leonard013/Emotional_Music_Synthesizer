"""
Train an LSTM music generator in PyTorch on the MAESTRO dataset.

Usage:
    python train_model.py
    python train_model.py --num_files 500 --epochs 80 --hidden_size 256
"""

import argparse
import collections
import glob
import pathlib
import time

import numpy as np
import pandas as pd
import pretty_midi
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
SEQ_LENGTH = 25
VOCAB_SIZE = 128
KEY_ORDER = ['pitch', 'step', 'duration']


# ──────────────────────────── Model ────────────────────────────────

class MusicGenerator(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, vocab_size=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.pitch_head = nn.Linear(hidden_size, vocab_size)
        self.step_head = nn.Linear(hidden_size, 1)
        self.duration_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        h = lstm_out[:, -1, :]
        pitch = self.pitch_head(h)
        step = self.step_head(h)
        duration = self.duration_head(h)
        return pitch, step, duration


# ──────────────────────────── Data ─────────────────────────────────

def midi_to_notes(midi_file: str) -> pd.DataFrame:
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start
    for note in sorted_notes:
        notes['pitch'].append(note.pitch)
        notes['step'].append(note.start - prev_start)
        notes['duration'].append(note.end - note.start)
        prev_start = note.start
    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


class MusicDataset(Dataset):
    def __init__(self, all_notes: np.ndarray, seq_length=SEQ_LENGTH, vocab_size=VOCAB_SIZE):
        self.seq_length = seq_length
        n = len(all_notes)
        num_seqs = n - seq_length
        self.inputs = np.empty((num_seqs, seq_length, 3), dtype=np.float32)
        self.labels = np.empty((num_seqs, 3), dtype=np.float32)
        for i in range(num_seqs):
            seq = all_notes[i:i + seq_length].copy()
            seq[:, 0] /= vocab_size  # normalize pitch in input
            self.inputs[i] = seq
            self.labels[i] = all_notes[i + seq_length]
        print(f"Dataset: {num_seqs} sequences")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.from_numpy(self.inputs[idx]), torch.from_numpy(self.labels[idx])


# ──────────────────────────── Loss ─────────────────────────────────

def non_negative_mse(y_true, y_pred):
    """MSE with penalty for negative predictions."""
    mse = (y_true - y_pred) ** 2
    penalty = 10 * torch.clamp(-y_pred, min=0.0)
    return (mse + penalty).mean()


# ──────────────────────────── Train ────────────────────────────────

def download_maestro():
    data_dir = SCRIPT_DIR / 'data' / 'maestro-v2.0.0'
    if not data_dir.exists():
        import urllib.request, zipfile
        print("Downloading MAESTRO dataset...")
        url = 'https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip'
        zip_path = SCRIPT_DIR / 'data' / 'maestro-v2.0.0-midi.zip'
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(SCRIPT_DIR / 'data')
        zip_path.unlink()
    return data_dir


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_dir = download_maestro()
    filenames = sorted(glob.glob(str(data_dir / '**/*.mid*'), recursive=True))
    print(f"Found {len(filenames)} MIDI files")

    num_files = min(args.num_files, len(filenames))
    print(f"Loading {num_files} files...")

    all_notes = []
    for i, f in enumerate(filenames[:num_files]):
        try:
            all_notes.append(midi_to_notes(f))
        except Exception:
            continue
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{num_files}")

    all_notes = pd.concat(all_notes)
    print(f"Total notes: {len(all_notes):,}")

    train_notes = np.stack([all_notes[key] for key in KEY_ORDER], axis=1)
    dataset = MusicDataset(train_notes, SEQ_LENGTH, VOCAB_SIZE)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=4, pin_memory=True,
                            drop_last=True)

    model = MusicGenerator(hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: LSTM(hidden={args.hidden_size}) — {total_params:,} params")

    # Loss weights: pitch loss is much larger so we downweight it
    w_pitch, w_step, w_dur = 0.05, 1.0, 1.0

    best_loss = float('inf')
    patience_counter = 0
    save_path = SCRIPT_DIR / 'music_generator_pytorch.pt'

    for epoch in range(args.epochs):
        model.train()
        total_loss = total_p = total_s = total_d = 0
        n = 0
        t0 = time.time()

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            pitch_pred, step_pred, dur_pred = model(inputs)
            lp = F.cross_entropy(pitch_pred, labels[:, 0].long())
            ls = non_negative_mse(labels[:, 1:2], step_pred)
            ld = non_negative_mse(labels[:, 2:3], dur_pred)
            loss = w_pitch * lp + w_step * ls + w_dur * ld

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_p += lp.item()
            total_s += ls.item()
            total_d += ld.item()
            n += 1

        avg = total_loss / n
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"loss={avg:.4f} (p={total_p/n:.3f} s={total_s/n:.4f} d={total_d/n:.4f}) | "
              f"lr={lr:.6f} | {time.time()-t0:.0f}s")

        scheduler.step(avg)

        if avg < best_loss:
            best_loss = avg
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> saved (best={best_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs)")
                break

    print(f"Training complete. Best loss: {best_loss:.4f}")
    print(f"Model saved to: {save_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num_files', type=int, default=750)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--hidden_size', type=int, default=128)
    p.add_argument('--patience', type=int, default=8)
    train(p.parse_args())
