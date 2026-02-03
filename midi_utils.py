import collections

import numpy as np
import pandas as pd
import pretty_midi
from matplotlib import pyplot as plt
from typing import Optional

from config import KEY_ORDER, SAMPLING_RATE


def midi_to_notes(midi_file: str) -> pd.DataFrame:
    """Extract pitch, step, and duration from a MIDI file into a DataFrame."""
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes["pitch"].append(note.pitch)
        notes["start"].append(start)
        notes["end"].append(end)
        notes["step"].append(start - prev_start)
        notes["duration"].append(end - start)
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


def notes_to_midi(
    notes: pd.DataFrame,
    out_file: str,
    instrument_name: str,
    velocity: int = 100,
) -> pretty_midi.PrettyMIDI:
    """Convert a DataFrame of notes back to a MIDI file."""
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(instrument_name)
    )

    prev_start = 0
    for i, note in notes.iterrows():
        start = float(prev_start + note["step"])
        end = float(start + note["duration"])
        midi_note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(note["pitch"]),
            start=start,
            end=end,
        )
        instrument.notes.append(midi_note)
        prev_start = start

    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm


def display_audio(pm: pretty_midi.PrettyMIDI, seconds: int = 10):
    """Synthesize MIDI to a waveform array."""
    waveform = pm.fluidsynth(fs=SAMPLING_RATE)
    waveform_short = waveform[: seconds * SAMPLING_RATE]
    return waveform_short


def plot_piano_roll(notes: pd.DataFrame, count: Optional[int] = None):
    """Visualize notes as a piano roll plot."""
    if count:
        title = f"First {count} notes"
    else:
        title = "Whole track"
        count = len(notes["pitch"])
    plt.figure(figsize=(20, 4))
    plot_pitch = np.stack([notes["pitch"], notes["pitch"]], axis=0)
    plot_start_stop = np.stack([notes["start"], notes["end"]], axis=0)
    plt.plot(
        plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker="."
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Pitch")
    plt.title(title)
    plt.show()


def analyze_midi(notes_sequence):
    """Extract 9 statistical features from a note sequence for K-Means classification.

    Args:
        notes_sequence: array-like of (pitch, step, duration) tuples/rows.

    Returns:
        dict with avg/max/std of step, duration, and pitch.
    """
    steps = [note[1] for note in notes_sequence]
    durations = [note[2] for note in notes_sequence]
    pitches = [note[0] for note in notes_sequence]

    return {
        "avg_step": np.mean(steps),
        "max_step": np.max(steps),
        "std_step": np.std(steps),
        "avg_duration": np.mean(durations),
        "max_duration": np.max(durations),
        "std_duration": np.std(durations),
        "avg_pitch": np.mean(pitches),
        "max_pitch": np.max(pitches),
        "std_pitch": np.std(pitches),
    }


def analyze_midi_file(midi_file: str) -> dict:
    """Extract 9 statistical features directly from a MIDI file path.

    Used by the K-Means training pipeline.
    """
    pm = pretty_midi.PrettyMIDI(midi_file)

    notes = []
    for instrument in pm.instruments:
        for note in instrument.notes:
            notes.append(note)

    sorted_notes = sorted(notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start
    steps = []
    for note in sorted_notes:
        steps.append(note.start - prev_start)
        prev_start = note.start

    durations = [note.end - note.start for note in notes]
    pitches = [note.pitch for note in notes]

    return {
        "avg_step": np.mean(steps),
        "max_step": np.max(steps),
        "std_step": np.std(steps),
        "avg_duration": np.mean(durations),
        "max_duration": np.max(durations),
        "std_duration": np.std(durations),
        "avg_pitch": np.mean(pitches),
        "max_pitch": np.max(pitches),
        "std_pitch": np.std(pitches),
    }
