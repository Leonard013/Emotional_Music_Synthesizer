# CLAUDE.md — Emotional Music Synthesizer

## Project Overview

Emotional Music Synthesizer is a deep learning research project that generates classical piano music responsive to a user's detected emotional state. It combines computer vision (facial emotion detection), audio processing (voice-to-MIDI), K-Means clustering, and LSTM neural networks to produce emotionally-guided MIDI compositions.

Built by students at Sapienza University of Rome. Designed to run on **Google Colab**.

## Repository Structure

```
Emotional_Music_Synthesizer/
├── CLAUDE.md                                # This file
├── Emotional_Music_Synthesizer.ipynb        # Main notebook — full pipeline
├── kmeans_clustering.ipynb                  # K-Means emotion classifier training
├── model.pkl                                # Pre-trained K-Means model (k=2)
├── music_generator.h5                       # Pre-trained LSTM music generator
├── shape_predictor_68_face_landmarks.dat    # dlib 68-point facial landmark detector (96 MB)
└── docs/
    ├── README.md                            # Project documentation
    └── model architecture.jpg               # System architecture diagram
```

There are no separate Python source files — all code lives in the two Jupyter notebooks.

## Technology Stack

| Category | Technologies |
|----------|-------------|
| ML/DL | TensorFlow/Keras, scikit-learn |
| Audio/MIDI | pretty_midi, pydub, fluidsynth, basic-pitch |
| Computer Vision | OpenCV, dlib |
| Data | NumPy, Pandas, SciPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Environment | Google Colab (Python 3.x, GPU) |

**No `requirements.txt` or build system exists.** All dependencies are installed inline via `!pip install` within the notebooks. The project is notebook-only with no packaging configuration.

## End-to-End Pipeline

### Main Workflow (`Emotional_Music_Synthesizer.ipynb`)

1. **Voice Recording** — Records 10s of audio from browser microphone → `voice.wav`
2. **Voice-to-MIDI** — Converts audio to MIDI using basic-pitch → `voice_basic_pitch.mid`
3. **Emotion Detection** — Captures webcam photo, detects face via Haar Cascade, extracts 68 dlib landmarks, classifies smile (mouth aspect ratio < 0.2 → happy `+1`, else sad `-1`)
4. **Note Extraction** — Parses MIDI into `(pitch, step, duration)` tuples
5. **LSTM Generation** — Sliding window of 25 notes fed to LSTM; outputs next note predictions
6. **Emotion Modulation** — Blends LSTM logits (α=0.65) with K-Means emotion scores (β=0.35), then divides by temperature (2.0); adjusts step/duration via Gaussian sampling
7. **MIDI Synthesis** — Converts generated notes to MIDI file, synthesizes audio via FluidSynth → `output.mid`

### K-Means Training Workflow (`kmeans_clustering.ipynb`)

1. Loads MAESTRO v3.0.0 (2890 files) + VGMIDI (906 files)
2. Extracts 9 features per file: avg/max/std of pitch, step, duration
3. Trains K-Means with k=2 (happy/sad clusters)
4. Saves model → `model.pkl`

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

```
seq_length = 25          # Sliding window size
vocab_size = 128         # MIDI pitch range
batch_size = 64
learning_rate = 0.005
epochs = 50
alpha = 0.65             # Model logits weight in pitch blending
beta = 0.35              # Emotion score weight in pitch blending
temperature = 2.0        # Softmax temperature for generation
gamma_std = 0.12         # Step emotion modifier std
delta_std = 0.15         # Duration emotion modifier std
num_predictions = 120    # Notes generated per song
```

## Important Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `record(sec)` | Main notebook | Records audio from browser microphone |
| `is_smiling(landmarks)` | Main notebook | Classifies emotion from mouth aspect ratio |
| `midi_to_notes(midi_file)` | Main notebook | Extracts pitch/step/duration from MIDI |
| `notes_to_midi(notes)` | Main notebook | Converts note DataFrame back to MIDI |
| `create_sequences()` | Main notebook | Builds sliding window training sequences |
| `non_negative_mse()` | Main notebook | Custom loss penalizing negative predictions |
| `predict_next_note()` | Main notebook | Generates next note with emotion modulation |
| `emotion_classifier()` | Main notebook | K-Means emotion scoring |
| `pitch_logits_emotion_encloser()` | Main notebook | Blends emotion score with LSTM logits |
| `step_emotion_encloser()` | Main notebook | Samples emotion-based step modifier |
| `duration_emotion_encloser()` | Main notebook | Samples emotion-based duration modifier |
| `analyze_midi(notes_sequence)` | Main notebook | Extracts 9 statistical features from a note sequence |
| `display_audio(pm)` | Main notebook | Synthesizes and plays MIDI via FluidSynth |
| `plot_piano_roll(notes)` | Main notebook | Visualizes notes as a piano roll plot |
| `take_photo(filename)` | Main notebook | Captures webcam photo for emotion detection |

## Development Notes

- **MAESTRO version mismatch**: The main notebook downloads MAESTRO v2.0.0 for LSTM training, while the K-Means notebook uses MAESTRO v3.0.0. These are different dataset versions.

- **Execution environment**: Google Colab is required due to webcam/microphone JavaScript integration and library compatibility.
- **No tests**: There is no formal test suite. Validation is done through training loss plots and listening to generated audio.
- **No CI/CD**: No automated pipelines.
- **Retraining is optional**: Pre-trained model files (`model.pkl`, `music_generator.h5`) are committed. The LSTM training cells are effectively skipped during normal use.
- **Large binary**: `shape_predictor_68_face_landmarks.dat` is 96 MB. It is a standard dlib asset and should not be modified.
- **MAESTRO dataset**: Not included in the repo; downloaded at runtime from Google's storage (~5 GB).

## Conventions for AI Assistants

1. **All code lives in notebooks** — edits should target `.ipynb` cells, not standalone `.py` files, unless restructuring is explicitly requested.
2. **Preserve notebook cell structure** — the notebooks are designed to run top-to-bottom sequentially. Avoid reordering cells.
3. **Hyperparameters are inline** — there is no config file. All tunable values are set directly in notebook cells.
4. **Binary model files** — do not regenerate or modify `model.pkl`, `music_generator.h5`, or `shape_predictor_68_face_landmarks.dat` unless the user specifically requests retraining.
5. **Emotion is binary** — the system uses only two emotion classes (happy/sad). Expanding this requires changes to both the K-Means model and the generation logic.
6. **Custom loss must be registered** — when loading `music_generator.h5`, pass `custom_objects={'mse_with_positive_pressure': non_negative_mse}` to `load_model()`.
7. **No dependency management** — if adding new libraries, add `!pip install` commands in the appropriate notebook cells.

## References

- Ferreira & Whitehead (2019): "Learning to Generate Music with Sentiment"
- Hawthorne et al. (2018): MAESTRO Dataset
- VGMIDI dataset for video game music emotion labeling
