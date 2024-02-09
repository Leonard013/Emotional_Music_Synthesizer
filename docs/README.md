# Emotional Music Synthesiser
## Introduction

Emotional Music Synthesizer is a Deep Learning model leveraging Long Short-Term Memory (LSTM) networks to generate classical music that resonates with the listener's emotional states.
## Dataset
At the heart of the Music Generator project is the goal to automate music composition, making it accessible to individuals without formal musical training. The system uses a collection of piano MIDI files from the [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro) to train an LSTM model. This model learns the patterns and structures of musical compositions, enabling it to predict and generate new musical sequences.


## Deep Learning Models and Techniques
- **Voice Recording/Music File**: initial input file the model will begin the output track with.
- **Features Extractor**: face and emotion recognition given a picture of the user's face.
- **Emotion Score**: Kmeans-based predictor trained to classify songs by emotion.
- **LSTM Networks**: main part, predicts new notes according to the previous ones.
- **Emotion Encloser**: incorporates emotion score to predict the new note.
- **List of Notes**: output list of notes.
<p align="center">
  <img src="model architecture.jpg" width="480px" />
</p>

## Approach
- The model takes as first input a voice recording or a music file and extracts the notes from it and an image of the user from which it derives the label belonging to {-1, 1} according to the emotion                      
Each note is a triple (pitch, step, duration) where pitch is the "tone" of the note, step is the distance from the previous one and duration is how much the note lasts.
- The window of notes (which will change over time after every prediction) is given to the lstm and its output to 3 different FFNN to predict pitch, step and duration as following:                                           
**pitch** (categorical cross entropy loss) outputs logits for each possible note (128)                
**step** (non-negative mse) predicts the most probable step                              
**duration** (non-negative mse) predicts the most probable duration                        
- Each obtained value is affected by the emotion score defined as:
<h3>  $$\text{emotion score}(x_i,l) = 1-\frac{e^{||x_i - \mu_l||}}{\sum_{j=1}^{2} e^{||x_j-\mu_l||}}$$  </h3>
where:
$l$ is the detected emotion <br />
$\mu_l$ is the cluster corresponding to the detected emotion <br />
In this way the score of a point is higher if it is closer to the correct emotion cluster and pitch logits are modified, we also decided to add some controlled randomness so to not incur into expectable and repeated sequences. 
<h3> $$logits  = \frac{\alpha \cdot emotion\_score + \beta \cdot logits}{temperature}$$ </h3>
with $\alpha = \beta = 0.5$ and $temperature = 2$ <br />
Finally a note is sampled according to the probabilty deriving from the normalization of latter scores <br />
For step and duration instead we empirically noticed that this 2 values are commonly higher in songs labeled as "sad" and lower in the "happy" therefore we decided to sample 2 parameters $\gamma$ and $\delta$ from 2 normal distribution
<h3> $$\mathbb{P}(\Gamma = \gamma) = N(\mu_{step}, \sigma_{step})$$ $$\mathbb{P}(\Delta = \delta) = N(\mu_{duration}, \sigma_{duration})$$ </h3>
where
<h3> $\mu_{step} = \mu_{duration} = 1$ </h3>
<h3> $\sigma_{step} = \sigma_{duration} = 0.1$ </h3>
then:
<h3> $$\gamma = 1-|\gamma|$$ $$\delta = 1-|\delta|$$ </h3>
Finally:
<h3> $$step = step_{pred} \cdot ( 1- l \cdot  \gamma)$$ $$duration = duration_{pred} \cdot ( 1- l \cdot  \delta)$$ </h3>
where:
$step_{pred}$ and $duration_{pred}$ are the step and duration from the the FFNN and $l \in$ { -1, 1 }
 
## Installation

To replicate the Music Generator system, users will need to set up a Python environment and download the required datasets. The project utilizes libraries such as `pretty_midi` for MIDI file manipulation and `tensorflow` for model training and inference.

## Quick Start

After setting up the environment and datasets, users can train their model using the provided scripts. The training process involves parsing MIDI files to extract musical notes, training the RNN model on this data, and then using the model to generate new music based on initial note sequences.


## Authors and Acknowledgements

We're a group of three Applied Computer Science and Artificial Intelligence students at Sapienza University of Rome, this is a project that we have undertaken as part of our Deep Learning Exam.

For any clarifications or further information, please feel free to contact us.

## License

The project is open-source, encouraging further experimentation and development in the field of AI generative models.

