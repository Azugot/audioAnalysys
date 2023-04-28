
import os
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

# Path to the audio files
CAPUCHIN_FILE = os.path.join(
    'data', 'Parsed_Capuchinbird_Clips', 'XC3776-3.wav')
NOT_CAPUCHIN_FILE = os.path.join(
    'data', 'Parsed_Not_Capuchinbird_Clips', 'afternoon-birds-song-in-forest-0.wav')


def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

def preprocess(file_path, label): 
    wav = load_wav_16k_mono(file_path)
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label


wave = load_wav_16k_mono(CAPUCHIN_FILE) #wave function of the capuchin bird
nwave = load_wav_16k_mono(NOT_CAPUCHIN_FILE)#wave function withou the capuchin bird

#ploting the waves using matplotlib
"""plt.plot(wave) 
plt.plot(nwave)
plt.show()
"""

#Reading the files from the folders
POS = os.path.join('data', 'Parsed_Capuchinbird_Clips')
NEG = os.path.join('data', 'Parsed_Not_Capuchinbird_Clips')

#Creating the datasets
pos = tf.data.Dataset.list_files(POS, '\*.wav')
neg = tf.data.Dataset.list_files(NEG, '\*.wav')

#Adding the labels to the datasets
positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))

#Concatenating the datasets
data = positives.concatenate(negatives)

#Shuffling the data
data = data.shuffle(buffer_size=1024)

#Calculate the wave length cycle
lenghts = []
for file in os.listdir(os.path.join('data', 'Parsed_Capuchinbird_Clips')):
    tensor_wave = load_wav_16k_mono(os.path.join('data', 'Parsed_Capuchinbird_Clips', file))
    lenghts.append(len(tensor_wave))

#Calculate min mean and max
tf.math.reduce_min(lenghts)
tf.math.reduce_mean(lenghts)
tf.math.reduce_max(lenghts)

filepath, label = positives.shuffle(buffer_size=10000).as_numpy_iterator().next()

spectrogram = preprocess(filepath, label)

plt.figure(figsize=(30,20))
plt.imshow(tf.transpose(spectrogram)[0])
plt.show()