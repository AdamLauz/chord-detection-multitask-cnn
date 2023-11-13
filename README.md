# chord-detection-multitask-cnn
In this project I've built a model that can perform 3 tasks using raw audio sounds:
1. Chord root identification 
2. Chord inversion identification (4 inversions)
3. Chord quality identification (minor, major, dim, major7 and so on)

# Dataset
I generated the dataset using the code under create_chords_dataset.py file.
A chord is sampled and converted to midi codes corresponding the pitches of the chord notes. The sound of the chord is rendered using VST3 plugin.
There are several virtual musical instruments used: piano, clarinet, acoustic and electric guitar, mandolin, bass guitar, strings and so on. Each instrument is assigned to a MIDI channel.
The possible traid and seventh chords are generated for each instrument according to its pitch range. A downsampling can be applied during this process. 

# Features
For each chord audio file, constant-Q transforms (CQT) are computed.
CQTs are saved alongside with the three labels (chord root, inversion and quality).

# Model
A multi-task convolutional neural network (CNN) model is used.
