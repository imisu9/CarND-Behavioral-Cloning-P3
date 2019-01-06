'''
Read in input files
'''

import os
import csv

samples =[]
validation_ratio = 0.2

with open('./driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    samples.append(line)
    
from sklearn.model_selection import train_test_split

train_sample, validation_sample = train_test_split(samples, test_size=validation_ratio)

'''
Define 'generator' function
'''

import cv2
import numpy as np
import sklearn

batch_size = 32

def generator(samples, batch_size=batch_size):
  num_samples = len(samples)
  while 1:  # Loop forever so the generator never terminates
    shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]
      
      images = []
      angles = []
      for batch_samples in batch_samples:
        name = './IMG/ + batch_sample[0].split('/')[-1]
        center_image = cv2.imread(name)
        center_angle = float(batch_sample[3])
        images.append(center_image)
        angles.append(center_angle)
      
      # Trim image to only see section with road
      X_train = np.array(images)
      y_train = np.array(angles)
      yield sklearn.utils.shuffle(X_train, y_train)
      
# Complie and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

'''
Employ NVIDIA CNN model described in End to End Learning for Self-Driving Cars

Model Architecture
'''

from keras.layers import Input, Lambda
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D
import tensorflow as tf

# Trimmed image format
ch, row, col = 3, 66, 200

model = Sequential()
# Normalize: preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(ch, row, col), output_shape=(ch, row, col)))
# Convolutional layer: 5x5 kernel, 24@31x98
model.add(Conv2D(24, 5, 5, strides=(2,2), padding='valid'))
# Convolutional layer: 5x5 kernel, 36@14x47
model.add(Conv2D(36, 5, 5, strides=(2,2), padding='valid'))
# Convolutional layer: 5x5 kernel, 48@5x22
model.add(Conv2D(48, 5, 5, strides=(2,2), padding='valid'))
# Convolutional layer: 3x3 kernel, 64@3x30
model.add(Conv2D(64, 3, 3, padding='valid'))
# Convolutional layer: 3x3 kernel, 64@1x18
model.add(Conv2D(64, 3, 3, padding='valid'))
# Flatten layer: 1164 neurons
model.add(Flatten())
# Fully connected layer: 100 neurons
model.add(Dense(100, activation='tanh')
# Fully connected layer: 50 neurons
model.add(Dense(50, activation='tanh')
# Fully connected layer: 10 neurons
model.add(Dense(10, activation='tanh')
# Fully connected output lyaer
model.add(Dense(1))

# Complie the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metircs=['accuracy'])

# Check the summary of this new model to confirm the architecture
model.summary()

from keras.utils import plot_model
plot_model(model, to_file='./examples/model.png')

checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True)
stopper = EarlyStopping(monitor='val_acc', min_delta=0.0003, patience=5)
history = model.fit_generator(train_generator,
                              steps_per_epoch=len(train_sample),
                              validation_data=validation_generator,
                              validation_steps=len(validation_samples),
                              epochs=10,
                              verbose=1,
                              callbacks=[checkpoint, stopper])

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model.save('model.h5')
