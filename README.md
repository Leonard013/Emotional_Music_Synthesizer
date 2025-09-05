# Whistle-to-Classical Music Synthesizer

A PyTorch-based system that analyzes whistled melodies and generates classical music using a Transformer architecture trained on the MAESTRO dataset.

## Features

- **Whistle Analysis**: Extract pitch, rhythm, and musical features from whistled audio
- **Transformer Architecture**: State-of-the-art music generation using attention mechanisms
- **MAESTRO Dataset**: Trained on high-quality classical piano performances
- **Real-time Processing**: Convert whistles to classical music in real-time

## Project Structure

```
├── src/
│   ├── whistle_analysis/     # Audio preprocessing and feature extraction
│   ├── data_processing/      # MAESTRO dataset handling
│   ├── models/              # Transformer architecture
│   ├── training/            # Training pipeline
│   └── inference/           # Real-time inference
├── configs/                 # Configuration files
├── data/                    # Data storage
│   ├── maestro/            # MAESTRO dataset
│   ├── whistles/           # Input whistle recordings
│   └── generated/          # Generated music output
├── notebooks/              # Jupyter notebooks for exploration
├── scripts/                # Utility scripts
└── tests/                  # Unit tests
```

## Setup

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate whistle-music-synthesizer
```

### 2. Download MAESTRO Dataset

```bash
# Download MAESTRO v3.0.0 (101GB)
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip
unzip maestro-v3.0.0.zip -d data/maestro/
```

### 3. Install Additional Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python scripts/train.py --config configs/training_config.yaml
```

### Inference

```bash
python scripts/inference.py --whistle_path data/whistles/example.wav --output_path data/generated/output.mid
```

### Real-time Processing

```bash
python scripts/realtime.py --input_device 0 --output_device 1
```

## Configuration

Edit `configs/training_config.yaml` to adjust:
- Model architecture parameters
- Training hyperparameters
- Data preprocessing settings
- Output configurations

## Dataset

This project uses the [MAESTRO Dataset](https://magenta.withgoogle.com/datasets/maestro) by Google Magenta, which contains ~200 hours of virtuosic piano performances with fine alignment between MIDI and audio.

## License

This project is licensed under the MIT License. The MAESTRO dataset is available under CC BY-NC-SA 4.0.
