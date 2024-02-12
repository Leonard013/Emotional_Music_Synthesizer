# Emotional Music Synthesizer
## Introduction

Emotional Music Synthesizer is a Deep Learning model leveraging Long Short-Term Memory (LSTM) networks to generate classical music that resonates with the listener's emotional states.
<br>
This software is intended for execution on Colab, ensuring accessibility to all users, regardless of the power of their machine and avoiding compatibility problems among the used libraries.
The model begins by recording the listener's voice, which initiates the composition process. Then, it captures a photo of the listener, and based on his/her emotional state, it adjusts the melody to either a happier or sadder tone. Initially, our intention was for the model to modify the composition in real time, reacting to the listener's changing emotions. However, due to the limitations of our computer's performance, we opted to base the composition on a single captured emotion instead.

## Dataset and Training
The goal of the Music Generator project is to automate music composition, making it either accessible to individuals without formal musical training or as a helpful tool for more expert people. The system uses a collection of piano MIDI files from the [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro) to train an LSTM model. This model learns the patterns and structures of musical compositions, enabling it to predict and generate new musical sequences.
The model has already been trained and it is possible to import the weights or start a new training, surely with a longer training the model will perform better but due to our computational power limitations (and colab limited gpu load time) we had to reduce it. The training is done running 50 epochs for 100 music files.

## Quick Start
To run the program it is necessary to run the colab notebook "Emotional_music_synthesizer.ipynb" in which step by step the whole generation process is illustrated. <br/>
Before running the file it is required to add to colab's directory file "shape_predictor_68_face_landmarks.dat" (face feature descriptor), "model.pkl" (k-means model) and "music_generator.h5" (music generator weights).
Due to the difficulty of finding stable, working libraries for the voice recording to use in GoogleColab, there might be difficulties in recording the initial audio in midi format. We suggest you to make it hear a recorded audio, to speak very loud or in alternative to give it directly a classical music midi file.

## Deep Learning Models and Techniques
- **Voice Recording/Music File**: initial input file the model will begin the output track with.
- **Features Extractor**: face and emotion recognition given a picture of the user's face.
- **Emotion Score**: Kmeans-based predictor trained to classify songs by emotion.
- **LSTM Networks**: main part, predicts new notes according to the previous ones.
- **FFNN**: positioned after lstm's output predict the most probable pitch step and duration of the next note.
- **Emotion Encloser**: incorporates emotion score to predict the new note.
- **List of Notes**: output list of notes.
<p align="center">
  <img src="model architecture.jpg" width="480px" />
</p>

## Approach
- The model takes as first input a voice recording or a music file and extracts the notes from it and an image of the user from which it derives the label belonging to {-1, 1} according to the emotion                      
Each note is a triple (pitch, step, duration) where pitch is the "tone" of the note, step is the time elapsed from the previous one's start and duration is how much the note lasts.
- The window of notes (which will change over time after every prediction) is given to the lstm and its output to 3 different FFNN to predict pitch, step and duration as following:                                           
**pitch** (categorical cross entropy loss) outputs logits for each possible note (128)                
**step** (non-negative mse) predicts the most probable step                              
**duration** (non-negative mse) predicts the most probable duration                        
- Each obtained value is affected by the emotion score defined as:
<h3>  $$\text{emotion score}(x_i,l) = 1-\frac{e^{||x_i - \mu_l||}}{\sum_{j=1}^{2} e^{||x_j-\mu_l||}}$$  </h3>
where:
<p>$l$ is the detected emotion</p> <br />
<p>$\mu_l$ is the cluster corresponding to the detected emotion </p><br/>
In this way the score of a point is higher if it is closer to the correct emotion cluster and pitch logits are modified, we also decided to add some controlled randomness so to not incur into expectable and repeated sequences. 
<h3> $$logits  = \frac{\alpha \cdot emotion\_score + \beta \cdot logits}{temperature}$$ </h3>
<p>with $\alpha = 0.65$ and $\beta = 0.35$ and $temperature = 2$</p> <br/>
Finally a note is sampled according to the probabilty deriving from the normalization of latter scores <br />
For step and duration, instead, we empirically noticed that this 2 values are commonly higher in songs labeled as "sad" and lower in the "happy" therefore we decided to sample 2 parameters $\gamma$ and $\delta$ from 2 normal distribution
<h3> $$\mathbb{P}(\Gamma = \gamma) = N(\mu_{step}, \sigma_{step})$$ $$\mathbb{P}(\Delta = \delta) = N(\mu_{duration}, \sigma_{duration})$$ </h3>
where
<h3> $\mu_{step} = \mu_{duration} = 1$ </h3>
<h3> $\sigma_{step} = 0.12$ and $\sigma_{duration} = 0.1$ </h3>
then:
<h3> $$\gamma = 1-|\gamma|$$ $$\delta = 1-|\delta|$$ </h3>
Finally:
<h3> $$step = step_{pred} \cdot ( 1- l \cdot  \gamma)$$ $$duration = duration_{pred} \cdot ( 1- l \cdot  \delta)$$ </h3>
where:
<p>$step_{pred}$ and $duration_{pred}$ are the step and duration from the the FFNN and $l \in$ { -1, 1 }</p>

## Authors and Acknowledgements

We're a group of three Applied Computer Science and Artificial Intelligence students at Sapienza University of Rome, this is a project that we have undertaken as part of our Deep Learning Exam.

For any clarifications or further information, please feel free to contact us.

## License

The project is open-source, encouraging further experimentation and development in the field of AI generative models.

## Citations

@article{ferreira_ismir_2019,
  title={Learning to Generate Music with Sentiment},
  author={Ferreira, Lucas N. and Whitehead, Jim},
  booktitle = {Proceedings of the Conference of the International Society for Music Information Retrieval},
  series = {ISMIR'19},
  year={2019},
}

@inproceedings{
  hawthorne2018enabling,
  title={Enabling Factorized Piano Music Modeling and Generation with the {MAESTRO} Dataset},
  author={Curtis Hawthorne and Andriy Stasyuk and Adam Roberts and Ian Simon and Cheng-Zhi Anna Huang and Sander Dieleman and Erich Elsen and Jesse Engel and Douglas Eck},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://openreview.net/forum?id=r1lYRjC9F7},
}
