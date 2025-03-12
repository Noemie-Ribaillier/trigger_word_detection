# Trigger word detection

## Project description
In this project we are going to carry on speech recognition. We are going to construct a speech dataset and implement an algorithm to do trigger word detection (also called keyword detection or wake word detection). Trigger word detection is the technology that allows devices like Amazon Alexa, Google Home and Apple Siri to wake up upon hearing a certain word. 
For this project, our trigger word will be "activate". Every time it hears the person saying "activate", it will make a chime sound. 


## WAV files
When working with speech recognition, we usually work with WAV files because:
* Lossless audio files: WAV files contain all the original audio data without any compression that reduces quality. WAV files store the raw audio data, preserving every detail of the sound, which is essential for accurate speech recognition. WAV files offer a more faithful reproduction of the original sound, providing the speech recognition system with the highest-quality input.
* Consistent sampling rate: the sampling rate (eg, 44.1 kHz or 16 kHz) and bit depth are clearly defined and consistent in WAV files. This makes it easier to pre-process and analyze the audio without needing to worry about variations in the encoding.
* Real-time processing and performance: WAV files are much easier to work with in terms of real-time processing since they don't require decompression before analysis, which can save time in speech recognition tasks. 


## Audio recording
A microphone records little variations in air pressure over time. It is these little variations in air pressure that the ear perceives as sound. The audio recording is like a long list of numbers measuring the little air pressure changes detected by the microphone. In this project, we will use audio sampled at 44100 Hz. This means that the microphone gives 44100 numbers per second. Thus, a 10 second audio clip is represented by 441000 numbers. 


## Divide audio into time-intervals
We may divide a 10 seconds interval of time with different units (steps):
* Raw audio divides 10 seconds into 441000 units (audio sample of 44100Hz * 10s)
* A spectrogram divides 10 seconds into 5511 units (Tx = 5511)
* Python library pydub synthesizes audio, and it divides 10 seconds into 10000 units.
* The output of our model will divide 10 seconds into 1375 units (Ty = 1375). For each of the 1375 time steps, the model predicts whether someone recently finished saying the trigger word "activate". 
All of these values are hyperparameters and can be changed (except the 441000, which is a function of the microphone) but we'll stick to them since we will use a model pre-trained with these values.


## Spectrogram
The spectrogram tells how much different frequencies are present in an audio clip at any moment in time. The graph represents how active each frequency is (y axis) over a number of time-steps (x axis). The color in the spectrogram shows the degree to which different frequencies are present (loud) in the audio at different points in time:
* Yellow means a certain frequency is more active or more present in the audio clip (louder)
* Blue denotes less active frequencies


## Dataset

### Training set
We need to create a dataset (for the trigger word detection algorithm) because it's hard to find a sample. As for every projects, the dataset should be as close as possible to the application we will want to run it on. In this case, we want to detect the word "activate" in working (noisy) environments (library, home, offices, open-spaces etc). So we are going to create recordings with a mix of positive words ("activate") and negative words (random words other than "activate") on different background sounds.
It's hard to aquire and label speech data. So the easiest is to gather background noises (libraries, cafes, restaurants, homes, offices etc) [being 10s clips], as well as snippets of audio of people saying positive/negative words (1 word per audio) and to synthesize our training data.

To synthesize a single training example, we:
* Pick a random 10 second background audio clip
* Pick an integer between 0 and 4 and randomly insert these audio clips of "activate" into the picked 10s clip
* Pick an integer between 0 and 2 and randomly insert these audio clips of negative words into the picked 10s clip
* Add each word audio clip on top of the background audio (so in the end, the audio keeps being 10s)
* Make sure no positive/negative clip is overlapping another one
* Label the positive ("activate") word as 1: y{t}=1 when the person just finished saying "activate" (and we set it up for the next 49 steps) or until the max size, otherwise y=0 (by default). Updating 50 steps can make the training data more balanced (and let more chances to discover the postive/trigger word).


### Validation set
To test the model we get a sample as similar as possible to the test set (so we use real audio).
We use 25 recorded 10-seconds audio clips of people saying "activate" and other random words, labeled by hand. 


## Model

### Gated Recurrent Unit (GRU)
GRU is a type of RNN, similar to LSTM. Like LSTM, GRU is a sequential model allowing information to be selectively remembered or forgotten over time (allowing the model to remember important information while discarding irrelevant details). GRU has a simpler architecture than LSTM, with fewer parameters which makes it simpler and faster to be trained than a LSTM.
The main difference between LSTM and GRU is how they handle the memory cell state:
* In LSTM, the memory cell state is maintained separately from the hidden state and is updated using 3 gates (input, forget and output gates)
* In GRU, the memory cell state is replaced with a "candidate activation vector", which is updated using 2 gates: 
    * Reset gate (r): determines how much of the previous hidden state (h_t-1) to forget
    * Update gate (z): determines how much of the candidate activation vector to incorporate into the new hidden state

GRU has the following inputs and output:
* Inputs: h_t-1 and x_t
* Output: h_t (that will be used as input at time t+1)

GRU processes sequential data, one element at a time, updating its hidden state based on the current input and the previous hidden state. At each time step, the GRU computes a "candidate activation vector" that combines information from the input and the previous hidden state. This candidate vector is then used to update the hidden state for the next time step. 
Step by step GRU process:
1. Compute the reset gate using inputs: r_t = sigmoid(W_r * [h_t-1, x_t]), with W_r the weight matrix that is learned during the training. It outputs a vector of numbers between 0 and 1 (from the sigmoid activation function) that controls the degree to which the previous hidden state is "reset" at the current time step.
2. Compute the update gate using inputs: z_t = sigmoid(W_z * [h_t-1, x_t]), with W_z the weight matrix that is learned during the training. It outputs the vector of numbers between 0 and 1 (from the sigmoid activation function) that controls the degree to which the candidate activation vector is incorporated into the new hidden state.
3. Candidate activation vector is computed using the current input (x_t) and a modified version of the previous hidden state (h_t-1) that is "reset" by the reset gate: h_tilde_t = tanh(W_h * [r_t*h_t-1, x_t]), with W_h another weight matrix.
The candidate activation vector is a modified version of the previous hidden state that is "reset" by the reset gate and combined with the current input. It is computed using a tanh activation function that gives values between -1 and 1.
4. The new hidden state h_t is computed by combining the candidate activation vector (h_tilde_t) with previous hidden state (h_t-1), weighted by the update gate: h_t = (1-z_t) * h_t-1 + z_t * h_tilde_t

RNN can have vanishing gradient problem. GRU helps in solving this problem by using gates that regulate the flow of gradients during training ensuring that important information is preserved and that gradients do not shrink excessively over time. By using these gates, GRUs maintain a balance between remembering important past information and learning new, relevant data.


### Model built for this project
The network uses the following layers:
* Conv1D: extracts low-level features and then possibly generates an output of a smaller dimension. It helps to speed up the model because later the GRU has to process less featuers (only 1375 timesteps rather than 5511 timesteps here). We use Conv1D because we are in a sequential problem.
* BatchNormalization: it normalizes the output of the previous layer by subtracting the batch mean and dividing by the batch standard deviation. It then applies learned scaling (gamma) and shifting (beta) parameters to the normalized output to allow the model to retain the capacity to express the original distribution if necessary. It aims at stabilizing the model, reduce overfitting and helps for faster convergence. 
In Deep Learning, it's common practise to have a convolutional layer, then a BN layer and then the activation function (eg ReLU) because BN normalizes the output of the layer before the activation introduces non-linearity. BN vs Dropout:
    * Apply BN before Dropout in convolutional layers because BN stabilizes activations, making the subsequent Dropout operation more effective and maintaining stable learning
    * Apply Dropout before BN in recurrent layers. Dropout helps prevent overfitting by randomly dropping units, and BN can then stabilize the activations that result from the randomness introduced by Dropout, making learning more stable and efficient.
* ReLU activation: introduce non-linearity
* Dropout: it removes some part of the units/neurons (set them to 0) for regularization and to prevent overfitting
* GRU: GRU is a sequential model allowing information to be selectively remembered or forgotten over time, it solves vanishing and exploding gradients issues we may have with RNNs
* TimeDistributed: the Dense layer is applied separately to each time step, preserving the time dimension.
* Sigmoid activation function: because we are in a binary classification problem (0 for non trigger detection or 1 for trigger detection)

Trigger word detection model takes a long time to train. So we use a pre-trained model (that was trained during 3h on a GPU, with a large training set of about 4k examples). We block the weights of all the batchnormalization layers since those weights are already adapted to the pre-trained model's previous training.

We use precision and recall since accuracy is not a good metric for this task. Indeed, the labels are heavily skewed to 0.

We use an unidirectional RNN (rather than bidirectional RNN) because we consider that focusing on past context is more critical than future context for detecting the activation word. This simplification also reduces computational overhead, as the model does not process information in both directions. 


## Insert a chime to acknowledge the "activate"
After estimating the probability of having detected the word "activate" at each output step, we can trigger a chime sound to play when the probability is above a certain threshold (for 20 consecutive steps, to avoid the random error). We just want to chime once (for each positive word), so we chime when the probability is over a certain threshold and at most every 75 output steps (not to get 2 chimes for the same trigger word).


## References
This script is coming from the Deep Learning Specialization course. I enriched it to this new version.
