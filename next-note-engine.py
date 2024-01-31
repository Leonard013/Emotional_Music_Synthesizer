import numpy as np
# ASSUMPTIONS:
# notes_sequence: numpy array of the sequence of notes so far
# notes: numpy array of all 128 notes
def emotion_classifier(notes_sequence):
    # for now it's just to try things
    return notes_sequence[:,0,0]

def pitch_logits_emotion_encloser(notes_sequence, pitches, pitch_logits, step, duration, label):
    # to tune hyperparameters
    alpha = 0.5
    beta = 0.5
    # everything happens after the prediction of the net
    # first we compute the new step and duration
    step = step_emotion_encloser(step, label)
    duration = duration_emotion_encloser(duration, label)
    # now we vectorize and do computations to output pitch logits influenced by the emotion
    vect_notes_sequence = np.array([notes_sequence] * 128)
    vect_step = np.array([step] * 128)
    vect_duration = np.array([duration] * 128)

    pitches = np.expand_dims(pitches, axis=1)

    vect_new_note = np.hstack((pitches, vect_step, vect_duration))
    vect_new_note = np.expand_dims(vect_new_note, axis=1)
    vect_notes_sequence = np.hstack((vect_notes_sequence, vect_new_note))
    emotion_score = emotion_classifier(vect_notes_sequence)
   
    logits = alpha*pitch_logits + beta*emotion_score

    return logits

def step_emotion_encloser(predicted_step, label):
    #label is assumed 1 or -1 according to the emotion
    # to tune hyperparameters
    gamma_mean = 0.0
    gamma_std = 1.0
    gamma = label*np.random.normal(gamma_mean, gamma_std, 1)
    return predicted_step + gamma

def duration_emotion_encloser(predicted_duration, label):
    #label is assumed 1 or -1 according to the emotion
    # to tune hyperparameters
    delta_mean = 0.0
    delta_std = 1.0
    delta = label*np.random.normal(delta_mean, delta_std, 1)
    return predicted_duration + delta

if __name__ == '__main__':
    # these should be the shapes of logits, notes, and notes sequence
    # in "predict_next_note"
    logits = np.expand_dims(np.random.randn(128), axis=0)
    pitches = np.random.randint(1, 100, size=128)
    sequence = np.random.rand(5, 3)
    step = np.random.rand()
    duration = np.random.rand()
    label = 1

    print(pitch_logits_emotion_encloser(sequence, pitches, logits, step, duration, label))