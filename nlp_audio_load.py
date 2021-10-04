import os
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras_preprocessing import image
import tensorflow as tf
import tensorflow_io as tfio
testing_wav_file_name  = 'data/no/0c5027de_nohash_0.wav'
# audio = tfio.audio.AudioIOTensor(testing_wav_file_name)
# audio_slice = audio[0:]
# # remove last dimension
# audio_tensor = tf.squeeze(audio_slice, axis=[-1])
# zero_padding = tf.zeros([16000] - tf.shape(audio_tensor), dtype=tf.float32)
# tensor = tf.cast(audio_tensor, tf.float32)
# equal_length = tf.concat([tensor, zero_padding], 0)
# spectrogram = tf.signal.stft(tensor, frame_length=255, frame_step=128)
# spectrogram = tf.abs(spectrogram)
# x = image.img_to_array(spectrogram)
# x = np.expand_dims(x, axis=0)
# images = np.vstack([x])
# model = tf.keras.models.load_model('nlp.h5')
# data = model.predict(images)
# print(data)
#
audio = tfio.audio.AudioIOTensor(testing_wav_file_name)
audio_slice = audio[0:]
audio_tensor = tf.squeeze(audio_slice, axis=-1)

def get_spectrogram(waveform):
  # Padding for files with less than 16000 samples
  zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
  # Concatenate audio with padding so that all audio clips will be of the
  # same length
  waveform = tf.cast(waveform, tf.float32)/10000.0
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)

  spectrogram = tf.abs(spectrogram)

  return spectrogram

spectrogram = get_spectrogram(audio_tensor)
images = np.array(spectrogram)
images = np.expand_dims(images, axis=0)
model = tf.keras.models.load_model('nlp.h5')
data = model.predict(images)
print(data)

