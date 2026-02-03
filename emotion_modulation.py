import pickle

import numpy as np
import scipy.special

from config import ALPHA, BETA, GAMMA_MEAN, GAMMA_STD, DELTA_MEAN, DELTA_STD, KMEANS_MODEL_PATH
from midi_utils import analyze_midi


def load_kmeans_model(path: str = KMEANS_MODEL_PATH):
    """Load the pre-trained K-Means emotion classifier."""
    with open(path, "rb") as f:
        return pickle.load(f)


def emotion_classifier(notes_sequence, current_emotion: int, kmeans_model) -> float:
    """Score a note sequence's alignment with the target emotion using K-Means.

    Args:
        notes_sequence: array of (pitch, step, duration) rows.
        current_emotion: +1 for happy, -1 for sad.
        kmeans_model: trained K-Means model.

    Returns:
        Emotion score (higher = closer to target emotion cluster).
    """
    notes_features = analyze_midi(notes_sequence)
    notes_features = np.array([value for value in notes_features.values()])
    notes_features = notes_features.reshape(1, -1)

    distances = kmeans_model.transform(notes_features.reshape(1, -1))
    softmax_distances = 1 - scipy.special.softmax(distances[0])
    emotion = 1 if current_emotion == 1 else 0

    emotion_score = softmax_distances[emotion]
    return emotion_score


def pitch_logits_emotion_encloser(
    notes_sequence, pitches, pitch_logits, step, duration, label, kmeans_model
):
    """Blend LSTM pitch logits with emotion score.

    logits = alpha * model_logits + beta * emotion_score
    where alpha=0.65 (model weight), beta=0.35 (emotion weight).
    """
    vect_notes_sequence = np.array([notes_sequence] * 128)
    vect_step = np.array([step] * 128)
    vect_duration = np.array([duration] * 128)

    pitches_col = np.expand_dims(pitches, axis=1)
    vect_new_note = np.hstack((pitches_col, vect_step, vect_duration))
    vect_new_note = vect_new_note.reshape((128, 1, 3))
    vect_notes_sequence = np.hstack((vect_notes_sequence, vect_new_note))
    emotion_score = emotion_classifier(vect_notes_sequence, label, kmeans_model)

    logits = ALPHA * pitch_logits + BETA * emotion_score
    return logits


def step_emotion_encloser() -> float:
    """Sample a Gaussian modifier for step adjustment."""
    gamma = abs(1 - np.random.normal(GAMMA_MEAN, GAMMA_STD, 1))
    return gamma


def duration_emotion_encloser() -> float:
    """Sample a Gaussian modifier for duration adjustment."""
    delta = abs(1 - np.random.normal(DELTA_MEAN, DELTA_STD, 1))
    return delta
