# import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# define constants
INPUT_SIZE = 64*24 # dimensions of the input data
DEGREE = 3 # degree of fourier series for seasonal inputs
LATENT_SIZE = 4*24 # the hours associated with each latent variable

data = pd.read_csv('data/final_dataset.csv', index_col=0, parse_dates=True)
print(data)

# Creating seasonal inputs using Fourier series
fourier = lambda x: np.stack([np.sin(2*np.pi*i*x) for i in range(1, DEGREE+1)] + [np.cos(2*np.pi*i*x) for i in range(1, DEGREE+1)], axis=-1)
starting_day = np.array(data.index.dayofyear)[:, np.newaxis] - 1
data_days = (starting_day + np.arange(0, INPUT_SIZE//24, LATENT_SIZE//24)) % 365
seasonal_data = fourier(data_days/365)
print(seasonal_data.shape)


# Split the data into training and testing sets
training_ratio = 0.8

train = data.values[:int(len(data)*training_ratio)]
test = data.values[int(len(data)*training_ratio):]
train_seasonal = seasonal_data[:int(len(data)*training_ratio)]
test_seasonal = seasonal_data[int(len(data)*training_ratio):]

# convert to tensors
train_tensor = tf.convert_to_tensor(train, dtype=tf.float32)
test_tensor = tf.convert_to_tensor(test, dtype=tf.float32)
train_seasonal_tensor = tf.convert_to_tensor(train_seasonal, dtype=tf.float32)
test_seasonal_tensor = tf.convert_to_tensor(test_seasonal, dtype=tf.float32)