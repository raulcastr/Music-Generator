<img src="https://ih1.redbubble.net/image.1790444441.2714/flat,128x,075,f-pad,128x128,f8f8f8.jpg" width="150" height="125"/>

# Music Generator

Raul Castrillo Martínez

## Content
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Future Work](#future-work)
- [Workflow](#workflow)
- [Links](#links)

## Project Description

The idea is to create a model that is able to learn from and predict the sequential structure of music and also be able to choose from a discrete set of possibilities for subsequent notes.


## Dataset

~ 70 midi songs from bach : http://www.jsbach.net/midi/


## Future Work

Create a model able to generate polyphonic music.

## Workflow

1 - Look for a proper dataset of songs:
To avoid chaotic outputs I decided to feed the model with music of the same style and author, in this case from the classical music author Johan Sebastian Bach.
As for  the format of the songs I used MIDI files because of it’s simplicity and low weight.
Also, I have limited myself to single instrument music (monophonic)  since this is my first cut model and I don’t have the privilege of time.
2 - Parsing:
We can’t simply feed the model with midi files because it wouldn’t understand it, instead we’ll use music21 library to parse the files and extract each note and it’s duration.
3 - Create the model:
Keras gives us the flexibility to be able to build a model that can handle the note and duration prediction simultaneously.
I used a Long short-term memory (LSTM) model that is a special kind of recurrent neural network , which is capable of learning long-term dependencies and it‘s able to recognise and encode long-term patterns.
4 - Training the model:
Be Aware that LSTM models take a lot time for training, in my case it took 8 to 9 hours to train 75 midi songs.
5 - Generate new songs:
Once the model is trained, it should be able to generate new music. It will try to predict notes based on the previous sequence using the notes and durations that it learned during the training.
6- Unparse the generated sequences: 
Now that the model has predicted some parsed data, the last step will be to unparse the notes with it’s durations and then convert it to a MIDI file so we can enjoy the results.


## Links

[Repository](https://github.com/raulcastr/Music-Generator/)  
[Slides](https://drive.google.com/file/d/1x0bZme_ojAbtKBni2b3bp6T5ckkuRTKu/view?usp=sharing)  
