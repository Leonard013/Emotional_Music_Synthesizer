"""Train the LSTM music generation model on the MAESTRO dataset.

Usage:
    python train_lstm.py [--epochs 50] [--num_files 750]

Downloads MAESTRO v2.0.0 if not already present, extracts notes,
builds sequences, trains the model, and saves to music_generator.h5.
"""

import argparse
import glob
import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from config import (
    SEED,
    SEQ_LENGTH,
    VOCAB_SIZE,
    BATCH_SIZE,
    EPOCHS,
    NUM_TRAIN_FILES,
    MAESTRO_URL,
    MAESTRO_DIR,
    MODEL_PATH,
    KEY_ORDER,
)
from model import build_model, create_sequences
from midi_utils import midi_to_notes


def main():
    parser = argparse.ArgumentParser(description="Train LSTM music generator")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--num_files", type=int, default=NUM_TRAIN_FILES)
    parser.add_argument("--output", type=str, default=MODEL_PATH)
    args = parser.parse_args()

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # Download MAESTRO dataset
    data_dir = pathlib.Path(MAESTRO_DIR)
    if not data_dir.exists():
        print("Downloading MAESTRO dataset...")
        tf.keras.utils.get_file(
            "maestro-v2.0.0-midi.zip",
            origin=MAESTRO_URL,
            extract=True,
            cache_dir=".",
            cache_subdir="data",
        )

    filenames = glob.glob(str(data_dir / "**/*.mid*"))
    print(f"Found {len(filenames)} MIDI files")

    # Extract notes
    print(f"Extracting notes from {args.num_files} files...")
    all_notes = []
    for f in filenames[: args.num_files]:
        notes = midi_to_notes(f)
        all_notes.append(notes)
    all_notes = pd.concat(all_notes)
    n_notes = len(all_notes)
    print(f"Total notes for training: {n_notes}")

    # Create dataset
    train_notes = np.stack([all_notes[key] for key in KEY_ORDER], axis=1)
    notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
    seq_ds = create_sequences(notes_ds, SEQ_LENGTH, VOCAB_SIZE)

    buffer_size = n_notes - SEQ_LENGTH
    train_ds = (
        seq_ds.shuffle(buffer_size)
        .batch(BATCH_SIZE, drop_remainder=True)
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    # Build and train model
    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="./training_checkpoints/ckpt_{epoch}",
            save_weights_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=5,
            verbose=1,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(train_ds, epochs=args.epochs, callbacks=callbacks)
    model.save(args.output)
    print(f"Model saved to {args.output}")

    # Plot loss
    plt.plot(history.epoch, history.history["loss"], label="total loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig("training_loss.png")
    print("Loss plot saved to training_loss.png")


if __name__ == "__main__":
    main()
