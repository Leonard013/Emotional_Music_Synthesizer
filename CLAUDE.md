# CLAUDE.md — Emotional Music Synthesizer

## Project Overview

Emotional Music Synthesizer is a deep learning research project that generates classical piano music responsive to a user's detected emotional state. It combines computer vision (facial emotion detection), audio processing (voice-to-MIDI), K-Means clustering, and LSTM neural networks to produce emotionally-guided MIDI compositions.

Built by students at Sapienza University of Rome.

## Repository Structure

```
Emotional_Music_Synthesizer/
├── main.py                                  # Entry point — CLI for full pipeline
├── config.py                                # All hyperparameters and paths
├── midi_utils.py                            # MIDI parsing, note extraction, synthesis
├── emotion_detection.py                     # Face detection, smile classification
├── model.py                                 # LSTM model definition, custom loss, sequences
├── emotion_modulation.py                    # K-Means emotion scoring, pitch/step/duration modulation
├── generate.py                              # Note generation loop with emotion guidance
├── train_lstm.py                            # Script to retrain LSTM on MAESTRO
├── train_kmeans.py                          # Script to retrain K-Means classifier
├── requirements.txt                         # Python dependencies
├── model.pkl                                # Pre-trained K-Means model (k=2)
├── music_generator.h5                       # Pre-trained LSTM music generator
├── shape_predictor_68_face_landmarks.dat    # dlib 68-point facial landmark detector (96 MB)
├── Emotional_Music_Synthesizer.ipynb        # Original notebook (legacy, kept for reference)
├── kmeans_clustering.ipynb                  # Original K-Means notebook (legacy, kept for reference)
└── docs/
    ├── README.md                            # Project documentation
    └── model architecture.jpg               # System architecture diagram
```

## Technology Stack

| Category | Technologies |
|----------|-------------|
| ML/DL | TensorFlow/Keras, scikit-learn |
| Audio/MIDI | pretty_midi, pydub, fluidsynth, basic-pitch, sounddevice |
| Computer Vision | OpenCV, dlib |
| Data | NumPy, Pandas, SciPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Environment | Python 3.x (local or Colab) |

Install dependencies: `pip install -r requirements.txt`

## Usage

```bash
# Full pipeline: record voice, detect emotion via webcam, generate music
python main.py

# Use pre-existing voice MIDI and specify emotion directly
python main.py --voice_midi voice_basic_pitch.mid --emotion happy

# Use a photo for emotion detection instead of webcam
python main.py --voice_midi voice_basic_pitch.mid --photo photo.jpg

# Adjust generation parameters
python main.py --voice_midi voice_basic_pitch.mid --emotion sad --num_notes 200 --temperature 1.5

# Retrain the LSTM model
python train_lstm.py --epochs 50 --num_files 750

# Retrain K-Means classifier
python train_kmeans.py --maestro_dir /path/to/maestro-v3.0.0 --vgmidi_dir /path/to/vgmidi/unlabelled
```

## End-to-End Pipeline

### Generation Workflow (`main.py` → `generate.py`)

1. **Voice Recording** — Records audio from microphone via `sounddevice` → `voice.wav`
2. **Voice-to-MIDI** — Converts audio to MIDI using `basic-pitch` → `voice_basic_pitch.mid`
3. **Emotion Detection** — Captures webcam photo (or uses provided photo/flag), detects face via dlib, classifies smile from mouth aspect ratio (< 0.2 → happy `+1`, else sad `-1`)
4. **Note Extraction** — Parses voice MIDI into `(pitch, step, duration)` tuples
5. **LSTM Generation** — Sliding window of 25 notes fed to LSTM; outputs next note predictions
6. **Emotion Modulation** — Blends LSTM logits (α=0.65) with K-Means emotion scores (β=0.35), then divides by temperature (2.0); adjusts step/duration via Gaussian sampling
7. **MIDI Synthesis** — Converts generated notes to MIDI file → `output.mid`

### K-Means Training Workflow (`train_kmeans.py`)

1. Loads MIDI files from MAESTRO and optionally VGMIDI directories
2. Extracts 9 features per file: avg/max/std of pitch, step, duration
3. Trains K-Means with k=2 (happy/sad clusters)
4. Saves model → `model.pkl`

### LSTM Training Workflow (`train_lstm.py`)

1. Downloads MAESTRO v2.0.0 if not present
2. Extracts notes from MIDI files, builds sliding-window sequences
3. Trains for 50 epochs with early stopping (patience=5)
4. Saves model → `music_generator.h5`

## Module Guide

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters, paths, and constants in one place |
| `midi_utils.py` | `midi_to_notes()`, `notes_to_midi()`, `analyze_midi()`, `analyze_midi_file()`, `display_audio()`, `plot_piano_roll()` |
| `emotion_detection.py` | `is_smiling()`, `detect_emotion_from_image()`, `detect_emotion_from_webcam()` |
| `model.py` | `build_model()`, `load_trained_model()`, `create_sequences()`, `non_negative_mse()` |
| `emotion_modulation.py` | `emotion_classifier()`, `pitch_logits_emotion_encloser()`, `step_emotion_encloser()`, `duration_emotion_encloser()`, `load_kmeans_model()` |
| `generate.py` | `predict_next_note()`, `generate_music()` |
| `train_lstm.py` | CLI script for LSTM training |
| `train_kmeans.py` | CLI script for K-Means training |
| `main.py` | CLI entry point orchestrating the full pipeline |

## Models

### LSTM Music Generator (`music_generator.h5`)
- **Input**: `(batch, 25, 3)` — 25 notes × 3 features (pitch/step/duration)
- **Architecture**: LSTM(128) → three heads: pitch Dense(128)+relu (logits, no softmax), step Dense(1)+relu, duration Dense(1)+relu
- **Parameters**: 84,354
- **Training**: 750 MAESTRO files, 50 epochs, batch_size=64, lr=0.005
- **Losses**: pitch=sparse categorical cross-entropy with `from_logits=True` (weight 0.05), step/duration=custom non-negative MSE (weight 1.0)
- **Custom loss** (`non_negative_mse`): MSE + 10× penalty for negative predictions

### K-Means Emotion Classifier (`model.pkl`)
- **Clusters**: 2 (happy vs sad)
- **Features**: 9 per MIDI file (avg/max/std of pitch, step, duration)
- **Training data**: 3,796 MIDI files

## Key Code Patterns

### Note Representation
Notes are stored as `(pitch: int, step: float, duration: float)` tuples or DataFrame rows. Pitch is MIDI note number (0-127), step is seconds since previous note start, duration is note length.

### Emotion Label Convention
```python
label = +1  # Happy (smiling)
label = -1  # Sad (not smiling)
```
Modulates step/duration: `value × (1 - label × modifier)` — happy shortens, sad lengthens.

### Sliding Window Generation
The LSTM generates one note at a time using a window of the previous 25 notes. After each prediction, the oldest note is removed and the new prediction is appended.

### Emotion Score Blending
```python
# In pitch_logits_emotion_encloser():
logits = 0.65 * model_logits + 0.35 * emotion_score

# Then in predict_next_note(), temperature is applied separately:
pitch_logits /= temperature  # temperature = 2.0
pitch = tf.random.categorical(pitch_logits, num_samples=1)
```
Emotion score comes from `1 - softmax(K-Means cluster distances)` for the target emotion cluster.

## Key Hyperparameters

All defined in `config.py`:

```
SEQ_LENGTH = 25          # Sliding window size
VOCAB_SIZE = 128         # MIDI pitch range
BATCH_SIZE = 64
LEARNING_RATE = 0.005
EPOCHS = 50
ALPHA = 0.65             # Model logits weight in pitch blending
BETA = 0.35              # Emotion score weight in pitch blending
TEMPERATURE = 2.0        # Softmax temperature for generation
GAMMA_STD = 0.12         # Step emotion modifier std
DELTA_STD = 0.15         # Duration emotion modifier std
NUM_PREDICTIONS = 120    # Notes generated per song
SMILE_THRESHOLD = 0.2    # Mouth aspect ratio threshold
```

## Development Notes

- **MAESTRO version mismatch**: The LSTM training uses MAESTRO v2.0.0, while the original K-Means notebook trained on MAESTRO v3.0.0. The `train_kmeans.py` script accepts any directory path.
- **No tests**: There is no formal test suite. Validation is done through training loss plots and listening to generated audio.
- **No CI/CD**: No automated pipelines.
- **Retraining is optional**: Pre-trained model files (`model.pkl`, `music_generator.h5`) are committed and used by default.
- **Large binary**: `shape_predictor_68_face_landmarks.dat` is 96 MB. It is a standard dlib asset and should not be modified.
- **MAESTRO dataset**: Not included in the repo; downloaded at runtime by `train_lstm.py`.
- **Original notebooks**: The `.ipynb` files are kept for reference but the `.py` files are the primary codebase.

## Conventions for AI Assistants

1. **Edit `.py` files** — all logic lives in the Python modules. The notebooks are legacy reference only.
2. **Hyperparameters live in `config.py`** — do not scatter magic numbers across modules.
3. **Binary model files** — do not regenerate or modify `model.pkl`, `music_generator.h5`, or `shape_predictor_68_face_landmarks.dat` unless the user specifically requests retraining.
4. **Emotion is binary** — the system uses only two emotion classes (happy/sad). Expanding this requires changes to both the K-Means model and the generation logic.
5. **Custom loss must be registered** — when loading `music_generator.h5`, use `model.load_trained_model()` which handles the custom objects mapping.
6. **Import from config** — all constants should be imported from `config.py`, not redefined locally.
7. **Dependencies** — if adding new libraries, add them to `requirements.txt`.

## References

- Ferreira & Whitehead (2019): "Learning to Generate Music with Sentiment"
- Hawthorne et al. (2018): MAESTRO Dataset
- VGMIDI dataset for video game music emotion labeling
