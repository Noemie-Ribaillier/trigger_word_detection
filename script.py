###########################################################################################################
#####                                                                                                 #####
#####                                     TRIGGER WORD DETECTION                                      #####
#####                                     Created on: 2025-02-28                                      #####
#####                                     Updated on: 2025-03-12                                      #####
#####                                                                                                 #####
###########################################################################################################

###########################################################################################################
#####                                            PACKAGES                                             #####
###########################################################################################################

# Clear the environment
globals().clear()

# Load the libraries
import numpy as np
from pydub import AudioSegment
import pygame
import random
import sys
import io
import glob
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, TimeDistributed, Conv1D
from tensorflow.keras.layers import GRU, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
from datetime import datetime

# Set up the right directory
import os
os.chdir('C:/Users/Admin/Documents/Python Projects/trigger_word_detection')
from general_functions import *


###########################################################################################################
#####                          LISTEN TO THE DIFFERENT TYPES OF AUDIO FILES                           #####
###########################################################################################################

# Initialize the mixer module
pygame.mixer.init()

# Load and play an audio file saying the word "activate"
pygame.mixer.music.load("raw_data/activates/1.wav")
pygame.mixer.music.play()

# Load and play an audio file saying a negative word (random word other than "activate")
pygame.mixer.music.load("raw_data/negatives/4.wav")
pygame.mixer.music.play()

# Load and play a background audio file (10s clips of background noise in different environments)
pygame.mixer.music.load("raw_data/backgrounds/1.wav")
pygame.mixer.music.play()


###########################################################################################################
#####                            FROM AN AUDIO RECORDING TO A SPECTROGRAM                             #####
###########################################################################################################

# The spectrogram tells us how much different frequencies are present in an audio clip at any moment in time. 

# Load and play an example containing several different words (included "activate" and negative words) and background noise
pygame.mixer.music.load("examples/example_train.wav")
pygame.mixer.music.play()

# Take a look at the spectrogram
x = graph_spectrogram("examples/example_train.wav")
plt.show()
# This graph represents how active each frequency is (y axis) over a number of time-steps (x axis)

# Open and read the data
_, data = wavfile.read("examples/example_train.wav")
print("Time steps in audio recording before spectrogram", data[:,0].shape)
print("Time steps in input after spectrogram", x.shape)

# Define the number of time steps input to the model from the spectrogram
Tx = x.shape[1]#5511
# Define the number of frequencies input to the model at each time step of the spectrogram
n_freq = x.shape[0]#101
# Define the number of time steps in the output of our model
Ty = 1375 


###########################################################################################################
#####                               GENERATE A SINGLE TRAINING EXAMPLE                                #####
###########################################################################################################

# Load the 3 type of audio files using pydub 
activates, negatives, backgrounds = load_raw_audio('raw_data/')

# Print the length of background, it should be 10k since it's a 10s file
print(len(backgrounds[0]))
# Print the length of the 1st activate file (it should be around 1k since 1s is needed to say this word)
print(len(activates[0]))
# Print the length of the 2nd activate file, still around 1k since 1s is needed to say a word (but can be different)
print(len(activates[1]))

# Create the function to retrieve a random time segment onto which we can insert an audio clip of duration segment_ms
def get_random_time_segment(segment_ms):
    """
    Get a random time segment of duration segment_ms in a 10000 ms audio clip
    
    Arguments:
    segment_ms -- the duration of the audio clip in ms
    
    Returns:
    segment_time -- a tuple (segment_start, segment_end) in ms
    """
    # Pick a random integer between 0 and the end - the length of the audio segment (to be sure the segment doesn't run past the 10sec background)
    # high parameter: is not included
    segment_start = np.random.randint(low=0, high=10000-segment_ms)  

    # Compute the segment end (-1 to align with the inclusive range)
    segment_end = segment_start + segment_ms - 1
    
    return (segment_start, segment_end)


# Create a function to check if a time segment overlaps with existing segments (because we don't want)
def is_overlapping(segment_time, previous_segments):
    """
    Check if the time of a segment overlaps with the times of existing segments
    
    Arguments:
    segment_time -- a tuple (segment_start, segment_end) for the new segment
    previous_segments -- a list of tuples (segment_start, segment_end) for the existing segments
    
    Returns:
    True if the time segment overlaps with any of the existing segments, False otherwise
    """
    # Get the start and end time from the segment_time
    segment_start, segment_end = segment_time
    
    # Initialize overlap as a "False" flag
    overlap = False
    
    # Loop over the previous_segments start and end times
    for previous_start, previous_end in previous_segments:
        # Compare start/end times and set the flag to True if there is an overlap
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True
            # After detecting the overlap, immediately exit the loop (no need to continue checking other conditions)
            break

    return overlap


# Control if those segments are overlapping or not (2 examples)
overlap1 = is_overlapping((950, 1430), [(2000, 2550), (260, 949)])
print(overlap1)
overlap2 = is_overlapping((2305, 2950), [(824, 1532), (1900, 2305), (3424, 3656)])
print(overlap2)


# Create the function to insert a new audio segment at a random time in the background audio
def insert_audio_clip(background, new_audio_segment, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step, ensuring that the 
    audio segment does not overlap with existing segments
    
    Arguments:
    background -- a 10s background audio recording
    new_audio_segment -- the audio clip to be inserted
    previous_segments -- times where audio segments have already been placed
    
    Returns:
    new_background -- the updated background audio
    """
    # Get the duration of the audio clip to be inserted in ms
    segment_ms = len(new_audio_segment)
    
    # Pick a random time segment onto which to insert the new audio clip
    segment_time = get_random_time_segment(segment_ms)
    
    # Check if the new segment_time overlaps with one of the previous_segments. If so, keep 
    # picking new segment_time at random until it doesn't overlap. To avoid an endless loop we retry 5 times
    retry = 5 
    while is_overlapping(segment_time, previous_segments) and retry >= 0:
        segment_time = get_random_time_segment(segment_ms)
        retry = retry - 1

    # If a segment_time is not overlapping, insert it to the background
    if not is_overlapping(segment_time, previous_segments):
        # Append the new segment_time to the list of previous_segments
        previous_segments.append(segment_time)
        # Superpose new audio segment and background (to keep having a background of 10s)
        new_background = background.overlay(new_audio_segment, position = segment_time[0])
    # Otherwise, keep the same background and input "fake/useless values" as segment_time
    else:
        new_background = background
        segment_time = (10000, 10000)

    return new_background, segment_time


# Insert randomly the first activate word to the first background clip
np.random.seed(21)
audio_clip, segment_time = insert_audio_clip(backgrounds[0], activates[0], [(3790, 4400)])
audio_clip.export("examples/insert_test.wav", format="wav")
print(segment_time)
pygame.quit()

# Load and play the audio file we just created (saying the word "activate")
pygame.init()
pygame.mixer.music.load("examples/insert_test.wav")
pygame.mixer.music.play()

# Create the function to insert 1 into the label vector y after the word "activate"
def insert_ones(y, segment_end_ms):
    """
    Update the label vector y. 
    The label of the 50 output steps strictly after the end of the segment should be set to 1 
    (meaning that the label of segment_end_y should be 0 while, the 50 following labels should be 1)
    
    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the example
    segment_end_ms -- the end time of the segment in ms
    
    Returns:
    y -- updated labels
    """
    # Get Ty (length of the output)
    _, Ty = y.shape

    # Make the mapping between background length (10s so 10000) and length of total output Ty for the number segment_end (cross product)
    segment_end_y = int(segment_end_ms * Ty / 10000.0)
    
    if segment_end_y < Ty:
        # Add 1 to the correct indexes in the output label (y)
        for i in range(segment_end_y + 1, segment_end_y + 51):
            # Be sure that we don't exceed the 10s audio clip (otherwise index out of range error)
            if i < Ty:
                y[0, i] = 1
    
    return y


# Control for an example (reminder that Ty is not 10k)
arr1 = insert_ones(np.zeros((1, Ty)), 9700)
arr2 = insert_ones(arr1, 4251)
plt.plot(arr2[0,:])
plt.show()


###########################################################################################################
#####                               GENERATE SEVERAL TRAINING EXAMPLES                                #####
###########################################################################################################

# Create a function to create our own training example
def create_training_example(background, activates, negatives, Ty):
    """
    Create a training example with a given background, activate and negative audios
    
    Arguments:
    background -- a 10s background audio
    activates -- a list of audio segments of the word "activate"
    negatives -- a list of audio segments of random words (different from "activate")
    Ty -- number of time steps in the output

    Returns:
    x -- spectrogram of the training example
    y -- label at each time step of the spectrogram
    """
    # Initialize y (label vector) of zeros and shape (1, Ty)
    y = np.zeros((1,Ty))

    # Initialize segment times as empty list
    previous_segments = []
    
    # Select 0-4 random "activate" audio clips from the entire list of "activates" recordings
    number_of_activates = np.random.randint(0, 5)
    random_indexes = np.random.randint(len(activates), size = number_of_activates)
    random_activates = [activates[i] for i in random_indexes]
    
    # Loop over randomly selected "activate" clips and insert in background
    for random_activate in random_activates:
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        # Retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time[0], segment_time[1]
        # Insert labels in "y" at segment_end
        y = insert_ones(y, segment_end)

    # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
    number_of_negatives = np.random.randint(0, 3)
    random_indexes = np.random.randint(len(negatives), size = number_of_negatives)
    random_negatives = [negatives[i] for i in random_indexes]

    # Loop over randomly selected negative clips and insert in background
    for random_negative in random_negatives:
        # Insert the audio clip on the background 
        # No need to keep the segment time for negative audio clips since y label remains 0 for negative audio clips
        background, _ =  insert_audio_clip(background, random_negative, previous_segments)
    
    # Standardize the volume of the audio clip 
    background = match_target_amplitude(background, -20.0)

    # Export new training example (we put the date in the name, otherwise we had permission issues)
    background.export('output/'+datetime.today().strftime('%Y-%m-%d%H-%M-%S').replace('-','')+"_train.wav", format="wav")
    pygame.quit()
    
    # Get the list of files in the folder output
    files = [f for f in os.listdir('output') if os.path.isfile(os.path.join('output', f))]
    # Get the most recent file (that was just created, to get the spectogram)
    most_recent_file = sorted(files, reverse=True)[0]

    # Get and plot spectrogram of the new recording (background with superposition of positive and negative audio clips)
    x = graph_spectrogram("output/"+most_recent_file)
    plt.show()

    return x, y


# Create 1 training example, taking the first background clip
np.random.seed(25)
x, y = create_training_example(backgrounds[0], activates, negatives, Ty)

# Listen to the training example we just created and compare it to the spectrogram generated above
pygame.init()
files = [f for f in os.listdir('output') if os.path.isfile(os.path.join('output', f))]
most_recent_file = sorted(files, reverse=True)[0]
pygame.mixer.music.load("output/"+most_recent_file)
pygame.mixer.music.play()

# Plot the associated labels for the generated training example (y=1 when the audio says "activate")
plt.plot(y[0])
plt.show()

# Generate a small sample
np.random.seed(4543)
nsamples = 32
X = []
Y = []
# Iterate on the nsamples, so we create nsamples samples
for i in range(nsamples):
    # Print i every 10 samples (to keep track of the created samples)
    if i%10 == 0:
        print(i)
    # Create an example (we have 2 backgrounds, so we switch from 1 to the other depending if i is even or odd)
    x, y = create_training_example(backgrounds[i % 2], activates, negatives, Ty)
    # Swap both axis and append to the list
    X.append(x.swapaxes(0,1))
    Y.append(y.swapaxes(0,1))
# Transform X and Y to array
X = np.array(X)
Y = np.array(Y)
# Get the length of X and Y
print(len(X))
print(len(Y))

# Save the data for further uses
np.save('XY/X_train.npy', X)
np.save('XY/Y_train.npy', Y)


###########################################################################################################
#####                                         BUILD THE MODEL                                         #####
###########################################################################################################

# Create the function to implement the model: from a spectogram we get a signal when it detects the trigger word
def modelf(input_shape):
    """
    Function creating the model's graph in Keras from a spectogram
    
    Argument:
    input_shape -- shape of the model's input data 

    Returns:
    model -- Keras model instance
    """
    # Create the input for the Keras model
    X_input = Input(shape = input_shape)

    # Step 1: CONV layer
    # Add a Conv1D with 196 units, kernel size of 15 and stride of 4
    X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)
    # Batch normalization ()
    X = BatchNormalization()(X)
    # ReLu activation
    X = Activation('relu')(X)
    # Dropout (with probability 0.8)
    X = Dropout(rate=0.8)(X)

    # Step 2: 1st GRU Layer 
    # GRU (use 128 units and return the sequences)
    # return_sequences parameter: to ensure that all the GRU's hidden states are fed to the next layer
    # (particularly useful when we need to feed the output of this layer to another recurrent layer)
    X = GRU(units=128, return_sequences = True)(X)
    # Dropout (with probability 0.8)
    X = Dropout(rate=0.8)(X)
    # Batch normalization
    X = BatchNormalization()(X)

    # Step 3: 2nd GRU Layer (same specifications than the 1st GRU layer)
    # GRU (use 128 units and return the sequences)
    X = GRU(units=128, return_sequences = True)(X)
    # Dropout (with probability 0.8)
    X = Dropout(rate=0.8)(X)
    # Batch normalization
    X = BatchNormalization()(X)
    # Dropout (with probability 0.8)
    X = Dropout(rate=0.8)(X)

    # Step 4: time-distributed dense layer
    # TimeDistributed with sigmoid activation 
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X) 

    # Set up the model
    model = Model(inputs = X_input, outputs = X)
    
    return model  


# Create the model
model = modelf(input_shape = (Tx, n_freq))

# Print the model summary to keep track of the shapes, and parameters
model.summary()
# The output of the network is of shape (None, 1375, 1) while the input is (None, 5511, 101)
# The Conv1D has reduced the number of steps from 5511 to 1375


###########################################################################################################
#####                             LOAD THE TRAINING SET AND FIT THE MODEL                             #####
###########################################################################################################

# Load the training examples we created
X = np.load("XY/X_train.npy")
Y = np.load("XY/Y_train.npy")

# Trigger word detection takes a long time to train. 
# To save time, we load a pre-trained model (trained for about 3 hours on a GPU using the architecture we built above and a large training set of about 4000 examples)

# Load the weights of a pretrained-model
model.load_weights('models/model.h5')

# Block the weights of all the batch-normalization layers since we only fine-tune the pre-trained model
# ie batchNormalization layers' weights are already adapted to the pre-trained model's previous training
# If we train a new model from scratch, we don't run the following lines 
model.layers[2].trainable = False
model.layers[7].trainable = False
model.layers[10].trainable = False

# Use the Adam optimizer
# beta_1 comes from momentum and beta_2 comes from RMSProp
# These parameters are almost never tuned, usually fixed to 0.9 and 0.999
opt = Adam(learning_rate=1e-6, beta_1=0.9, beta_2=0.999)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy","precision","recall"])
# We use precision and recall since the labels are heavily skewed to 0s (so accuracy is not the best metric)

# Train the model
model.fit(X, Y, batch_size = 32, epochs=2)


###########################################################################################################
#####                            LOAD THE TEST SET AND EVALUATE THE MODEL                             #####
###########################################################################################################

# Load preprocessed dev set examples
X_dev = np.load("XY/X_dev.npy")
Y_dev = np.load("XY/Y_dev.npy")

# Evaluate the model on the dev set
loss, acc, precision, recall = model.evaluate(X_dev, Y_dev)
print("Dev set accuracy = ", np.round(acc,2), "Dev set precision = ", round(precision,2), "Dev set recall = ", round(recall,2))


###########################################################################################################
#####                        MAKE PREDICTIONS AND ADD CHIME ON "ACTIVATE" WORD                        #####
###########################################################################################################

# Create the function to detect the trigger word
def detect_triggerword(file):
    """
    Compute the probability that the audio mentions the trigger word at each time step
    Plot the spectogram of the audio file and the probability computed by the model that the trigger word is said
    
    Argument:
    file -- audio path

    Returns:
    predictions -- probabilities that the model detects the trigger word 
    """
    # Set up the plot window, with 2 plots on the same window (on top of each other)
    plt.subplot(2, 1, 1)
    
    # Read and load the audio file
    audio_clip = AudioSegment.from_wav(file)
    # Correct the amplitude/volume of the input file before prediction 
    audio_clip = match_target_amplitude(audio_clip, -20.0)

    # Plot the spectogram of the audio
    x = graph_spectrogram(file)

    # The spectrogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model, so we switch the dimensions
    x  = x.swapaxes(0,1)
    # Add a dimension at axis 0, meaning shape is (1, Tx, freqs), the batch dimension
    x = np.expand_dims(x, axis=0)
    # Predict the probability that the trigger word is said at every time step
    predictions = model.predict(x)

    plt.subplot(2, 1, 2)
    # Plot the predictions (only the 2nd dimension contains the predictions)
    plt.plot(predictions[0,:,0])
    plt.ylabel('Probability')
    plt.show()

    return predictions


# Create the function to export a new audio file, adding the chime when the model detects the word activate
chime_file = "raw_data/chime.wav"
def chime_on_activate(file, predictions, threshold):
    """
    Create a new audio file, adding the chime on the audio file when the model detects "activate" word
    The probability of detecting the trigger word has to be higher than a certain threshold for 20 consecutive timesteps
    There is a minimum of 75 time steps between 2 chimes (to avoid chiming twice for the same trigger word)
    
    Argument:
    file -- audio path
    predictions -- predictions array
    threshold -- probability threshold, from which we consider the model predicted the trigger word
    """
    # Read and load the audio file and the chime file (trigger noise)
    audio_clip = AudioSegment.from_wav(file)
    chime = AudioSegment.from_wav(chime_file)

    # Standardize both volumes (to the audio volume)
    chime = match_target_amplitude(chime,audio_clip.dBFS)

    # Get the length of the predictions (middle dimension contains the predictions)
    Ty = predictions.shape[1]

    # Initialize the number of consecutive and output steps to 0
    consecutive_timesteps = 0
    output_step = 0

    # Loop over the output steps in the y
    while output_step < Ty:
        # Increment consecutive output steps
        consecutive_timesteps += 1
        # If prediction is higher than the threshold for 20 consecutive output steps
        if consecutive_timesteps > 20:
            # Superpose audio and background using pydub (with cross product to determine the position since the audio last 10s so 10k steps but the predictions use Ty steps)
            audio_clip = audio_clip.overlay(chime, position = (output_step * (audio_clip.duration_seconds* 1000) / Ty))
            # Reset consecutive output steps to 0
            consecutive_timesteps = 0
            # Move output_step further away because we don't want to output 2 chimes for the same word 
            # We want min 75 steps between 2 chimes (- 20 for the consecutive_timesteps we have already done)
            output_step += (75-20)
            # Don't execute the next if (because with this loop we are already in the case where we have 20 consecutive timesteps, meaning 20 steps with a proba higher than threshold)
            continue
        # If the probability is smaller than the threshold reset the consecutive_timesteps counter
        if predictions[0, output_step, 0] < threshold:
            consecutive_timesteps = 0
        output_step += 1
    
    # Export the audio file with the chime
    audio_clip.export("output/"+datetime.today().strftime('%Y-%m-%d%H-%M-%S').replace('-','')+"_chime_output.wav", format='wav')


###########################################################################################################
#####                                       TEST ON DEV EXAMPLES                                      #####
###########################################################################################################

# Check the performance of our model on 2 unseen audio clips from the development set

# Load and listen to the 1st example
file1 = "raw_data/dev/1.wav"
pygame.init()
pygame.mixer.music.load(file1)
pygame.mixer.music.play()

# Run the model on the audio clip and control if it adds a chime after "activate"
prediction = detect_triggerword(file1)
chime_on_activate(file1, prediction, 0.5)
files = [f for f in os.listdir('output') if os.path.isfile(os.path.join('output', f))]
most_recent_file = sorted(files, reverse=True)[0]
pygame.mixer.music.load("output/"+most_recent_file)
pygame.mixer.music.play()
pygame.quit()


# Load and listen to the 2nd example
file2  = "raw_data/dev/2.wav"
pygame.init()
pygame.mixer.music.load(file2)
pygame.mixer.music.play()

# Run the model on the audio clip and control if it adds a chime after "activate"
prediction = detect_triggerword(file2)
chime_on_activate(file2, prediction, 0.5)
files = [f for f in os.listdir('output') if os.path.isfile(os.path.join('output', f))]
most_recent_file = sorted(files, reverse=True)[0]
pygame.mixer.music.load("output/"+most_recent_file)
pygame.mixer.music.play()
pygame.quit()
