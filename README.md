# MAESTRO Transformer Continuation

This repository contains a single Jupyter notebook, `maestro_transformer.ipynb`, that demonstrates how to train a compact Transformer on the [MAESTRO](https://magenta.tensorflow.org/datasets/maestro) piano performance dataset and autoregressively continue music after a two-second seed.

## Contents
- `maestro_transformer.ipynb`: End-to-end workflow for preprocessing MAESTRO MIDI files, training the model, and generating continuations.
- `.gitignore`: Ignores local data, MIDI exports, and notebook checkpoints.
- `README.md`: You are here.

## Quick start
1. **Install dependencies** (Python ≥ 3.10 recommended):
   ```bash
   pip install torch pretty_midi tqdm
   ```
2. **Download MAESTRO v3.0.0 MIDI files** and extract them into a folder named `maestro/` next to the notebook (or set `MAESTRO_ROOT=/path/to/extracted`).
3. **Launch Jupyter** and open the notebook:
   ```bash
   jupyter notebook maestro_transformer.ipynb
   ```
4. **Run the notebook** from top to bottom. Training defaults to a small subset of the dataset; adjust `MAX_FILES`, model size, or epochs as needed.

The final cell writes a generated MIDI continuation to disk so you can listen to the synthesized performance.
