# Emotional Music Synthesiser
## Introduction

Emotional Music Synthesizer is a Deep Learning model leveraging Long Short-Term Memory (LSTM) networks to generate classical music that resonates with the listener's emotional states.
## Dataset
At the heart of the Music Generator project is the goal to automate music composition, making it accessible to individuals without formal musical training. The system uses a collection of piano MIDI files from the [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro) to train an LSTM model. This model learns the patterns and structures of musical compositions, enabling it to predict and generate new musical sequences.

## Machine Learning Models and Techniques

- **Recurrent Neural Networks (RNNs)**: The core technology behind the music generation, capable of learning sequences and patterns in the musical data.
- **Data Preprocessing**: Techniques for extracting musical elements from MIDI files, such as pitch, step, and duration, which serve as inputs for the model.

<p align="center">
  <img src="model architecture.jpg" width="480px" />
</p>
## Installation

To replicate the Music Generator system, users will need to set up a Python environment and download the required datasets. The project utilizes libraries such as `pretty_midi` for MIDI file manipulation and `tensorflow` for model training and inference.

## Quick Start

After setting up the environment and datasets, users can train their model using the provided scripts. The training process involves parsing MIDI files to extract musical notes, training the RNN model on this data, and then using the model to generate new music based on initial note sequences.

## Visualization and Interaction

The project includes functionalities to visualize the generated music, converting MIDI sequences back into a more interpretable format. Additionally, it explores the impact of different initial conditions or parameters on the generated music, offering users a way to interact with and influence the composition process.

## Authors and Acknowledgements

The Music Generator project is a collaborative effort, aiming to push the boundaries of AI in creative domains. It represents a synthesis of music theory, machine learning, and software development, showcasing the potential of AI to innovate in the arts.

## License

The project is open-source, encouraging further experimentation and development in the field of AI-generated music. It exemplifies the collaborative spirit of the AI and music communities, inviting contributions and improvements.

