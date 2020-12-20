#!/usr/bin/env python
# coding: utf-8

# In[44]:


import os
import numpy as np
import music21
import glob
import pickle
from keras import layers

import keras.backend as K 
from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils import np_utils


# In[1]:


def read_midi_parsed(seq_len = 32,mode = "read"):
    notes = []
    durations = []
    
    if mode=="read":
        path_list = glob.glob(os.path.join("../data/training_songs/", "*.mid"))
        print(len(path_list), 'files in total')

        for i, file in enumerate(path_list):
            print(i+1, "Parsing %s" % file)
            original_score = music21.converter.parse(file).chordify()

            for interval in range(1):

                score = original_score.transpose(interval)

                notes.extend(['START'] * seq_len)
                durations.extend([0]* seq_len)

                for element in score.flat:

                    if isinstance(element, music21.note.Note):
                        if element.isRest:
                            notes.append(str(element.name))
                            durations.append(element.duration.quarterLength)
                        else:
                            notes.append(str(element.nameWithOctave))
                            durations.append(element.duration.quarterLength)

                    if isinstance(element, music21.chord.Chord):
                        notes.append('.'.join(n.nameWithOctave for n in element.pitches))
                        durations.append(element.duration.quarterLength)

        
        pickle.dump(notes, open("../data/stored_data/parsed_data/notes", "wb"))
        pickle.dump(durations, open("../data/stored_data/parsed_data/durations", "wb"))
        
    else:
        notes = pickle.load(open("../data/stored_data/parsed_data/notes", "rb"))
        durations = pickle.load(open("../data/stored_data/parsed_data/durations", "rb"))
        
    return notes , durations


# In[46]:


def create_network(n_notes, n_durations, embed_size = 100, rnn_units = 256):
    """ create the structure of the neural network """
    # There are two inputs to the network: the sequence of previous note names and duration values. 
    notes_in = layers.Input(shape = (None,))
    durations_in = layers.Input(shape = (None,))
    
    # The Embedding layers convert the integer values of the note names and durations into vectors.
    x1 = layers.Embedding(n_notes, embed_size)(notes_in)
    x2 = layers.Embedding(n_durations, embed_size)(durations_in) 
    
    # The vectors are concatenated to form one long vector that will be used as input into the recurrent layers.
    x = layers.Concatenate()([x1,x2])
    
    # Two stacked LSTM layers are used as the recurrent part of the network. Notice how we set return_sequences to True to make 
    # each layer pass the full sequence of hidden states to the next layer, rather than just the final hidden state.
    x = layers.LSTM(rnn_units, return_sequences=True)(x)
    x = layers.LSTM(rnn_units, return_sequences=True)(x)

    # The alignment function is just a Dense layer with one output unit and tanh activation. We can use a Reshape layer to 
    # squash the output to a single vector, of length equal to the length of the input sequence (seq_length).
    e = layers.Dense(1, activation='tanh')(x)
    e = layers.Reshape([-1])(e)
    
    # The weights are calculated through applying a softmax activation to the alignment values.
    alpha = layers.Activation('softmax')(e)
    
    # To get the weighted sum of the hidden states, we need to use a RepeatVector layer to copy the weights rnn_units times
    # to form a matrix of shape [rnn_units, seq_length], then transpose this matrix using a Permute layer to get a matrix of 
    # shape [seq_length, rnn_units]. We can then multiply this matrix pointwise with the hidden states from the final LSTM layer,
    # which also has shape [seq_length, rnn_units]. Finally, we use a Lambda layer to perform the summation along the seq_length 
    # axis, to give the context vector of length rnn_units.
    alpha_repeated = layers.Permute([2, 1])(layers.RepeatVector(rnn_units)(alpha))
    c = layers.Multiply()([x, alpha_repeated])
    c = layers.Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(rnn_units,))(c)
    
    # The network has a double-headed output, one for the next note name and one for the next note length.
    notes_out = layers.Dense(n_notes, activation = 'softmax', name = 'pitch')(c)
    durations_out = layers.Dense(n_durations, activation = 'softmax', name = 'duration')(c)
   
    # The final model accepts the previous note names and note durations as input and outputs a distribution
    # for the next note name and next note duration.
    model = Model([notes_in, durations_in], [notes_out, durations_out])
   
    # The model is compiled using categorical_crossentropy for both the note name and note duration output heads, as this is a
    # multiclass classification problem.
    
    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], optimizer=RMSprop(lr = 0.001))

    return model


# In[42]:


def prepare_sequences(notes, durations, lookups, distincts, seq_len =32):
    """ Prepare the sequences used to train the Neural Network """

    note_to_int, int_to_note, duration_to_int, int_to_duration = lookups
    note_names, n_notes, duration_names, n_durations = distincts

    notes_network_input = []
    notes_network_output = []
    durations_network_input = []
    durations_network_output = []

    # create input sequences and the corresponding outputs
    for i in range(len(notes) - seq_len):
        notes_sequence_in = notes[i:i + seq_len]
        notes_sequence_out = notes[i + seq_len]
        notes_network_input.append([note_to_int[char] for char in notes_sequence_in])
        notes_network_output.append(note_to_int[notes_sequence_out])

        durations_sequence_in = durations[i:i + seq_len]
        durations_sequence_out = durations[i + seq_len]
        durations_network_input.append([duration_to_int[char] for char in durations_sequence_in])
        durations_network_output.append(duration_to_int[durations_sequence_out])

    n_patterns = len(notes_network_input)

    # reshape the input into a format compatible with LSTM layers
    notes_network_input = np.reshape(notes_network_input, (n_patterns, seq_len))
    durations_network_input = np.reshape(durations_network_input, (n_patterns, seq_len))
    network_input = [notes_network_input, durations_network_input]

    notes_network_output = np_utils.to_categorical(notes_network_output, num_classes=n_notes)
    durations_network_output = np_utils.to_categorical(durations_network_output, num_classes=n_durations)
    network_output = [notes_network_output, durations_network_output]

    return (network_input, network_output)


# In[43]:


def sample_with_temp(preds, temperature):

    if temperature == 0:
        return np.argmax(preds)
    else:
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        return np.random.choice(len(preds), p=preds)

