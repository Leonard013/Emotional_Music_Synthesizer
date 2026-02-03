# Hyperparameters and configuration

# Random seed
SEED = 42

# Audio
SAMPLING_RATE = 16000
RECORD_SECONDS = 10

# MIDI / Model
SEQ_LENGTH = 25
VOCAB_SIZE = 128
BATCH_SIZE = 64
LEARNING_RATE = 0.005
EPOCHS = 50
NUM_PREDICTIONS = 120

# Loss weights
LOSS_WEIGHTS = {
    "pitch": 0.05,
    "step": 1.0,
    "duration": 1.0,
}

# Emotion modulation
ALPHA = 0.65        # Model logits weight in pitch blending
BETA = 0.35         # Emotion score weight in pitch blending
TEMPERATURE = 2.0   # Softmax temperature for generation
GAMMA_MEAN = 1.0
GAMMA_STD = 0.12    # Step emotion modifier std
DELTA_MEAN = 1.0
DELTA_STD = 0.15    # Duration emotion modifier std

# Smile detection
SMILE_THRESHOLD = 0.2

# Paths
MAESTRO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip"
MAESTRO_DIR = "data/maestro-v2.0.0"
MODEL_PATH = "music_generator.h5"
KMEANS_MODEL_PATH = "model.pkl"
LANDMARK_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
VOICE_WAV_PATH = "voice.wav"
VOICE_MIDI_PATH = "voice_basic_pitch.mid"
PHOTO_PATH = "photo.jpg"
OUTPUT_MIDI_PATH = "output.mid"

# Key order for note features
KEY_ORDER = ["pitch", "step", "duration"]

# Number of MAESTRO files used for training
NUM_TRAIN_FILES = 750
