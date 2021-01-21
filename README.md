# Automatic Music Generator

Raul Castrillo Martínez


## Content
- [Project Description](#project-description)
- [Hypotheses / Questions](#hypotheses-questions)
- [Dataset](#dataset)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Workflow](#workflow)
- [Links](#links)

## Project Description
The idea is to create a model that is able to learn from and predict the sequential structure of music and also be able to choose from a discrete set of possibilities for subsequent notes.

## Hypotheses / Questions
Can we create music generation models using machine learning?

## Dataset

midi songs from bach : http://www.jsbach.net/midi/


## Conclusion

As for automatic music generation: Although there are a few impressive music generation models around the net capable of delivering amazing results, CLICK  I don’t think computers can compete against humans as for music composition, because in my opinion creativity is, and always will be, a human endeavor.

## Future Work

Create a model able to generate polyphonic music.

## Workflow

1 -
First of all I looked for a proper dataset of songs
To avoid chaotic outputs I decided to feed the model with music of the same style and author, in this case from the classical music author Johan Sebastian Bach.
As for  the format of the songs I used MIDI files because of it’s simplicity and low weight.
Also, I have limited myself to single instrument music (monophonic)  since this is my first cut model and I don’t have the privilege of time.
2 -
Once i had an appropriate dataset I had to parse it , we can’t simply feed the model with midi files because it wouldn’t understand it, instead we’ll use music21 library to parse the files  and extract each note and it’s duration.
3 -
My next step was to create a deep learning model using Keras and Tensorflow .
Keras gives us the flexibility to be able to build a model that can handle the note and duration prediction simultaneously.
I used a Long short-term memory model that is a special kind of recurrent neural network , which is capable of learning long-term dependencies and it‘s able to recognise and encode long-term patterns.
4 -
Now that i have the model and the parsed data it’s time to train the model.  Be Aware that LSTM models take a lot time for training, in my case it took 8 to 9 hours to train 75 midi songs.
5 -
Once the model is trained, it will be able to generate new music. It will try to predict notes based on the previous sequence. It will use the notes and durations that it learned during the training.
Now the model has predicted some parsed data, the last step will be to unparse the notes with it’s duration and then convert it to a MIDI file.


## Links

[Repository](https://github.com/raulcastr/Music-Generator-Project)  
[Slides](https://drive.google.com/file/d/1x0bZme_ojAbtKBni2b3bp6T5ckkuRTKu/view?usp=sharing)  
