"""
Generate piano music using a trained LSTM model.

Usage:
    python generate_music.py
    python generate_music.py --temperature 1.2 --num_notes 200
    python generate_music.py --seed_midi my_song.mid --output my_output.wav
"""

import argparse
import collections
import glob
import pathlib

import numpy as np
import pandas as pd
import pretty_midi
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
SEQ_LENGTH = 25
VOCAB_SIZE = 128
KEY_ORDER = ['pitch', 'step', 'duration']
SAMPLING_RATE = 16000


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


# ──────────────────────────── MIDI I/O ─────────────────────────────

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


def notes_to_midi(notes: pd.DataFrame, out_file: str,
                  instrument_name: str = 'Acoustic Grand Piano',
                  velocity: int = 100) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(instrument_name))
    prev_start = 0
    for _, note in notes.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])
        midi_note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(note['pitch']),
            start=max(0, start),
            end=max(start + 0.01, end),  # minimum duration to avoid silent notes
        )
        instrument.notes.append(midi_note)
        prev_start = start
    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm


# ──────────────────────────── Generation ───────────────────────────

@torch.no_grad()
def generate(model, seed_notes, num_notes=120, temperature=1.0):
    """Autoregressively generate notes from a seed sequence."""
    input_seq = seed_notes[:SEQ_LENGTH].copy()
    input_seq[:, 0] /= VOCAB_SIZE  # normalize pitch

    generated = []
    prev_start = 0

    for _ in range(num_notes):
        x = torch.from_numpy(input_seq).unsqueeze(0).float()
        pitch_logits, step_pred, dur_pred = model(x)

        # Sample pitch with temperature
        pitch_logits = pitch_logits.squeeze(0) / temperature
        pitch_probs = F.softmax(pitch_logits, dim=-1)
        pitch = torch.multinomial(pitch_probs, 1).item()

        # Step and duration (clamp to non-negative)
        step = max(0.0, step_pred.item())
        duration = max(0.0, dur_pred.item())

        start = prev_start + step
        end = start + duration
        generated.append((pitch, step, duration, start, end))
        prev_start = start

        # Shift window
        new_note = np.array([[pitch / VOCAB_SIZE, step, duration]])
        input_seq = np.vstack([input_seq[1:], new_note])

    return pd.DataFrame(generated, columns=[*KEY_ORDER, 'start', 'end'])


# ──────────────────────────── Seed ─────────────────────────────────

def get_seed_notes(seed_path=None):
    if seed_path:
        print(f"Seed: {seed_path}")
        return midi_to_notes(seed_path)

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

    filenames = sorted(glob.glob(str(data_dir / '**/*.mid*'), recursive=True))
    if not filenames:
        raise RuntimeError("No MIDI files found")
    # Pick a random seed file
    idx = np.random.randint(len(filenames))
    print(f"Seed: {pathlib.Path(filenames[idx]).name}")
    return midi_to_notes(filenames[idx])


# ──────────────────────────── Main ─────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Generate piano music with trained LSTM')
    p.add_argument('--temperature', type=float, default=1.0,
                   help='Sampling temperature (default: 1.0, lower=more conservative)')
    p.add_argument('--num_notes', type=int, default=120,
                   help='Number of notes to generate (default: 120)')
    p.add_argument('--seed_midi', type=str, default=None,
                   help='Seed MIDI file (default: random MAESTRO file)')
    p.add_argument('--output', type=str, default='output.wav',
                   help='Output WAV path')
    p.add_argument('--output_midi', type=str, default='output.mid',
                   help='Output MIDI path')
    p.add_argument('--model', type=str,
                   default=str(SCRIPT_DIR / 'music_generator_pytorch.pt'),
                   help='Path to model weights')
    p.add_argument('--hidden_size', type=int, default=128,
                   help='LSTM hidden size (must match trained model)')
    args = p.parse_args()

    # Load model
    model = MusicGenerator(hidden_size=args.hidden_size)
    model.load_state_dict(torch.load(args.model, map_location='cpu', weights_only=True))
    model.eval()
    print(f"Loaded model from {args.model}")

    # Get seed
    raw_notes = get_seed_notes(args.seed_midi)
    seed = np.stack([raw_notes[key] for key in KEY_ORDER], axis=1)
    print(f"Seed has {len(seed)} notes")

    # Generate
    print(f"Generating {args.num_notes} notes (temperature={args.temperature})...")
    generated = generate(model, seed, args.num_notes, args.temperature)
    print(generated.head(10).to_string())

    # Save MIDI
    out_pm = notes_to_midi(generated, args.output_midi)
    print(f"MIDI -> {args.output_midi}")

    # Synthesize WAV
    waveform = out_pm.fluidsynth(fs=SAMPLING_RATE)
    sf.write(args.output, waveform, SAMPLING_RATE)
    print(f"WAV  -> {args.output} ({len(waveform)/SAMPLING_RATE:.1f}s)")


if __name__ == '__main__':
    main()
