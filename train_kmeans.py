"""Train the K-Means emotion classifier on MIDI datasets.

Usage:
    python train_kmeans.py --maestro_dir PATH --vgmidi_dir PATH [--output model.pkl]

Loads MIDI files from MAESTRO and VGMIDI directories, extracts 9 features
per file, trains K-Means with k=2, and saves the model.
"""

import argparse
import glob
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from config import KMEANS_MODEL_PATH
from midi_utils import analyze_midi_file


def main():
    parser = argparse.ArgumentParser(description="Train K-Means emotion classifier")
    parser.add_argument("--maestro_dir", type=str, required=True,
                        help="Path to MAESTRO dataset directory")
    parser.add_argument("--vgmidi_dir", type=str, default=None,
                        help="Path to VGMIDI unlabelled directory (optional)")
    parser.add_argument("--output", type=str, default=KMEANS_MODEL_PATH)
    args = parser.parse_args()

    # Collect MIDI file paths
    files = []

    # MAESTRO files
    maestro_folders = [
        f for f in os.listdir(args.maestro_dir)
        if os.path.isdir(os.path.join(args.maestro_dir, f))
    ]
    for folder in maestro_folders:
        pattern = os.path.join(args.maestro_dir, folder, "*.midi")
        files.extend(glob.glob(pattern))
        pattern_mid = os.path.join(args.maestro_dir, folder, "*.mid")
        files.extend(glob.glob(pattern_mid))

    # VGMIDI files (optional)
    if args.vgmidi_dir:
        vgmidi_folders = [
            f for f in os.listdir(args.vgmidi_dir)
            if os.path.isdir(os.path.join(args.vgmidi_dir, f))
        ]
        for folder in vgmidi_folders:
            pattern = os.path.join(args.vgmidi_dir, folder, "*.mid")
            files.extend(glob.glob(pattern))

    print(f"Found {len(files)} MIDI files")

    # Extract features
    print("Extracting features...")
    train_dataset = np.empty((len(files), 9), dtype=float)
    errors = 0
    for i, f in enumerate(files):
        try:
            data = analyze_midi_file(f)
            train_dataset[i] = np.array(list(data.values()))
        except Exception as e:
            print(f"  Skipping {f}: {e}")
            errors += 1

    if errors > 0:
        train_dataset = train_dataset[: len(files) - errors]
        print(f"Skipped {errors} files with errors")

    print(f"Training K-Means on {len(train_dataset)} samples...")
    kmeans_model = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans_model.fit_predict(train_dataset)

    print(f"Cluster distribution: {np.bincount(labels)}")

    # Visualize
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(
        train_dataset[:, 3], train_dataset[:, 6],
        c=labels, cmap="viridis", s=50, alpha=0.5,
    )
    ax.scatter(
        kmeans_model.cluster_centers_[:, 3],
        kmeans_model.cluster_centers_[:, 6],
        c="red", s=50,
    )
    ax.set_xlabel("Average duration")
    ax.set_ylabel("Average pitch")
    plt.savefig("kmeans_clusters.png")
    print("Cluster plot saved to kmeans_clusters.png")

    # Save model
    with open(args.output, "wb") as f:
        pickle.dump(kmeans_model, f)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
