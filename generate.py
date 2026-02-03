import numpy as np
import pandas as pd
import tensorflow as tf

from config import (
    SEQ_LENGTH,
    VOCAB_SIZE,
    TEMPERATURE,
    NUM_PREDICTIONS,
    KEY_ORDER,
    MODEL_PATH,
    OUTPUT_MIDI_PATH,
)
from model import load_trained_model
from midi_utils import midi_to_notes, notes_to_midi, plot_piano_roll
from emotion_modulation import (
    load_kmeans_model,
    pitch_logits_emotion_encloser,
    step_emotion_encloser,
    duration_emotion_encloser,
)


def predict_next_note(pitches, notes, model, temperature, label, kmeans_model):
    """Generate a single note as (pitch, step, duration) using the LSTM and emotion modulation.

    Args:
        pitches: array of candidate pitch values (0-127).
        notes: current sliding window of shape (seq_length, 3).
        model: trained Keras LSTM model.
        temperature: softmax temperature for sampling.
        label: emotion label (+1 happy, -1 sad).
        kmeans_model: trained K-Means model.

    Returns:
        Tuple of (pitch, step, duration).
    """
    assert temperature > 0

    inputs = tf.expand_dims(notes, 0)
    predictions = model.predict(inputs, verbose=0)
    pitch_logits = predictions["pitch"]
    step = predictions["step"]
    duration = predictions["duration"]

    pitch_logits = pitch_logits_emotion_encloser(
        notes, pitches, pitch_logits, step, duration, label, kmeans_model
    )
    pitch_logits /= temperature

    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch)
    duration = tf.squeeze(duration)
    step = tf.squeeze(step)

    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)

    return int(pitch), float(step), float(duration)


def generate_music(
    voice_midi_path: str,
    label: int,
    model_path: str = MODEL_PATH,
    output_path: str = OUTPUT_MIDI_PATH,
    num_predictions: int = NUM_PREDICTIONS,
    temperature: float = TEMPERATURE,
    show_plot: bool = True,
) -> pd.DataFrame:
    """Run the full music generation pipeline.

    Args:
        voice_midi_path: path to the voice recording MIDI file.
        label: emotion label (+1 happy, -1 sad).
        model_path: path to trained LSTM .h5 file.
        output_path: path to write the generated MIDI file.
        num_predictions: number of notes to generate.
        temperature: softmax temperature for pitch sampling.
        show_plot: whether to display the piano roll plot.

    Returns:
        DataFrame of generated notes.
    """
    print("Loading models...")
    model = load_trained_model(model_path)
    kmeans_model = load_kmeans_model()

    print("Extracting notes from voice recording...")
    raw_notes = midi_to_notes(voice_midi_path)
    sample_notes = np.stack([raw_notes[key] for key in KEY_ORDER], axis=1)

    # Get instrument name from voice MIDI for output
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(voice_midi_path)
    instrument_name = pretty_midi.program_to_instrument_name(pm.instruments[0].program)

    # Initialize sliding window (pitch normalized by vocab_size)
    input_notes = sample_notes[:SEQ_LENGTH] / np.array([VOCAB_SIZE, 1, 1])

    generated_notes = []
    prev_start = 0
    pitches = np.arange(128)
    pitches = np.expand_dims(pitches, axis=1)

    gamma = step_emotion_encloser()
    delta = duration_emotion_encloser()
    print(f"Emotion modifiers: gamma={gamma}, delta={delta}")

    print(f"Generating {num_predictions} notes (label={label})...")
    for i in range(num_predictions):
        pitch, step, duration = predict_next_note(
            pitches, input_notes, model, temperature, label, kmeans_model
        )
        start = prev_start + step
        end = start + duration

        step = float(step * (1 - label * gamma))
        duration = float(duration * (1 - label * delta))

        input_note = (pitch, step, duration)
        generated_notes.append((*input_note, start, end))

        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(
            input_notes, np.expand_dims(input_note, 0), axis=0
        )
        prev_start = start

    generated_notes = pd.DataFrame(
        generated_notes, columns=(*KEY_ORDER, "start", "end")
    )

    print(f"Writing MIDI to {output_path}...")
    out_pm = notes_to_midi(generated_notes, out_file=output_path, instrument_name=instrument_name)

    if show_plot:
        plot_piano_roll(generated_notes)

    print("Done.")
    return generated_notes
