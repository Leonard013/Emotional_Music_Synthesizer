import tensorflow as tf

from config import SEQ_LENGTH, VOCAB_SIZE, LEARNING_RATE, LOSS_WEIGHTS, KEY_ORDER


def non_negative_mse(y_true: tf.Tensor, y_pred: tf.Tensor):
    """Custom loss: MSE + 10x penalty for negative predictions."""
    mse = (y_true - y_pred) ** 2
    positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
    return tf.reduce_mean(mse + positive_pressure)


def build_model() -> tf.keras.Model:
    """Build and compile the LSTM music generation model.

    Architecture:
        Input(seq_length, 3) -> LSTM(128) -> three heads:
            pitch: Dense(128, relu) - logits for categorical pitch
            step:  Dense(1, relu)   - non-negative time step
            duration: Dense(1, relu) - non-negative duration
    """
    input_shape = (SEQ_LENGTH, 3)
    inputs = tf.keras.Input(input_shape)
    x = tf.keras.layers.LSTM(128)(inputs)

    outputs = {
        "pitch": tf.keras.layers.Dense(128, activation="relu", name="pitch")(x),
        "step": tf.keras.layers.Dense(1, activation="relu", name="step")(x),
        "duration": tf.keras.layers.Dense(1, activation="relu", name="duration")(x),
    }

    model = tf.keras.Model(inputs, outputs)

    loss = {
        "pitch": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        "step": non_negative_mse,
        "duration": non_negative_mse,
    }

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(loss=loss, loss_weights=LOSS_WEIGHTS, optimizer=optimizer)
    return model


def load_trained_model(model_path: str) -> tf.keras.Model:
    """Load a previously trained model from an .h5 file.

    Handles both Keras 2 and Keras 3 .h5 formats. If direct loading
    fails (e.g. Keras 2 model on Keras 3), falls back to building a
    fresh model and loading weights only.
    """
    try:
        from keras.models import load_model

        return load_model(
            model_path,
            custom_objects={"mse_with_positive_pressure": non_negative_mse},
        )
    except (ValueError, TypeError):
        # Keras 2 .h5 file on Keras 3: build model and load weights
        model = build_model()
        model.load_weights(model_path)
        return model


def create_sequences(dataset: tf.data.Dataset, seq_length: int, vocab_size: int = VOCAB_SIZE) -> tf.data.Dataset:
    """Build sliding-window sequences from a note dataset.

    Each example is a window of seq_length notes (input) and the next note (label).
    Pitch values are normalized by vocab_size.
    """
    seq_length_plus = seq_length + 1

    windows = dataset.window(seq_length_plus, shift=1, stride=1, drop_remainder=True)
    flatten = lambda x: x.batch(seq_length_plus, drop_remainder=True)
    sequences = windows.flat_map(flatten)

    def scale_pitch(x):
        x = x / [vocab_size, 1.0, 1.0]
        return x

    def split_labels(sequences):
        inputs = sequences[:-1]
        labels_dense = sequences[-1]
        labels = {key: labels_dense[i] for i, key in enumerate(KEY_ORDER)}
        return scale_pitch(inputs), labels

    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)
