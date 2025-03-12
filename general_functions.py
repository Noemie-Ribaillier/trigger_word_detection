# Load the libraries
import matplotlib.pyplot as plt
# To work with WAV sound files
from scipy.io import wavfile
import os
# Manipulate audio files
from pydub import AudioSegment


# Create the function to open and get the informations of a wav file
def get_wav_info(wav_file):
    """
    Get the informations of a wav file
    
    Arguments:
    wav_file -- audio file (.wav format)
    
    Returns:
    rate -- sample rate of WAV file (number of numbers given by the microphone per second, number of hertz)
    data -- data read from a WAV file
    """
    # Open and get the informations of the wav file
    rate, data = wavfile.read(wav_file)

    return rate, data


# Create the function to compute and plot the spectrogram of a wav audio file
def graph_spectrogram(wav_file):
    """
    Compute and plot the spectrogram of a wav audio file
    The spectogram shows how the frequency content of a signal changes over time

    Arguments:
    wav_file -- audio file (.wav format)
    
    Returns:
    pxx -- power spectral density (how much power is present in the signal at each frequency component)
    """
    # Print the number of hertz and data of the wav_file
    rate, data = get_wav_info(wav_file)

    # Define the length of each window segment, number of points used in the Fast Fourier Transform (FFT)
    # It determines the frequency resolution of the Fourier transform (usually chosen as a power of 2 for efficiency)
    nfft = 200

    # Define the sampling frequencies (it represents how many samples per second are recorded from the continuous signal)
    fs = 8000

    # Define the overlap between windows, number of overlapping samples between consecutive frames (i.e., how much the analysis window shifts at each step)
    # Normally we set it up as nfft / 2 (but we'll use a model later that has been trained on 120)
    noverlap = 120

    # Define the number of channels
    nchannels = data.ndim

    # Compute and plot a spectrogram of data
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
        # pxx: 2D array where each entry represents the power (intensity) of the signal at a particular time and frequency (the values indicate how much signal energy is present at different frequency bands across time)
        # freqs: frequency bins for the spectrogram. It is a 1D array that holds the corresponding frequencies (in Hz) for each row in the pxx array (the length of this array corresponds to the number of frequency bins (which is determined by the nfft parameter and the sampling frequency fs)).
        # bins: represents the time bins or time steps. It is a 1D array that corresponds to the time windows at which the spectrogram was computed. Length bins computed as follow: (data.shape[0]-noverlap)/(nfft-noverlap)
    # The plot defines the axis as following:
    # * x-axis: from 0 to xxx (so here 0-55) representing the time. Each value corresponds to a time step in the signal, representing how the signal evolves over time
    # * y-axis: from 0 to fs/2 (so here 0-4000) representing the frequency. It goes until fs/2 because the Nyquist theorem states that the maximum frequency that can be resolved in the signal is half the sampling rate
    # for a specific x, the vertical bar corresponds to the frequency content of the signal at that particular time. Each value corresponds to a frequency in the signal, showing how much energy (or power) is present at that frequency.

    return pxx


# Create a function used to standardize volume of audio clip
def match_target_amplitude(sound, target_dBFS):
    """
    Adjust the loudness (volume) of an audio clip to a target level in dBFS
    dBFS stands for decibels relative to full scale (it's a unit of measurement used to express the volume of an audio signal in a digital format)
    If the dBFS of an audio is equal to -3, it means that the audio is 3 decibels below the maximum amplitude that the system can handle

    Arguments:
    sound -- an audio clip
    target_dBFS -- the desired volume level we want to achieve for the audio

    Returns:
    standardized_audio -- the audio with adjusted volume
    """
    # Compute the difference in dBFS between the target volume level and the current volume of the audio
    # It represents how much we need to increase or decrease the audio's volume in dB to reach the target volume
    change_in_dBFS = target_dBFS - sound.dBFS
    
    # Adjust the volume of the audio by the amount we just found/determined
    # (positive change_in_dBFS increases the volume while negative change_in_dBFS decreases the volume)
    standardized_audio = sound.apply_gain(change_in_dBFS)

    return standardized_audio


# Load read and store raw audio files for speech synthesis
def load_raw_audio(path):
    """
    Load, read and store raw audio files for each category (activate, negative and background)
    
    Arguments:
    path -- path of the audio folder
    
    Returns:
    activates -- store all the activate audios in a list
    negatives -- store all the negative audios in a list
    backgrounds -- store all the background audios in a list
    """
    # Create empty lists to gather activate, negative and background audios
    activates = []
    backgrounds = []
    negatives = []

    # Iterate on the files from "activates" folder
    for file in os.listdir(path + "activates"):
        # Take all audio files (with wav extension)
        if file.endswith("wav"):
            # Read and load the audio file
            activate = AudioSegment.from_wav(path + "activates/" + file)
            # Append each activate audio to the list
            activates.append(activate)

    # Iterate on the files from backgrounds folder
    for file in os.listdir(path + "backgrounds"):
        # Take all audio files (with wav extension)
        if file.endswith("wav"):
            # Read and load the audio file
            background = AudioSegment.from_wav(path + "backgrounds/" + file)
            # Append each background audio to the list
            backgrounds.append(background)

    # Iterate on the files from negatives folder
    for file in os.listdir(path + "negatives"):
        # Take all audio files (with wav extension)
        if file.endswith("wav"):
            # Read and load the audio file
            negative = AudioSegment.from_wav(path + "negatives/" + file)
            # Append each negative audio to the list
            negatives.append(negative)

    return activates, negatives, backgrounds