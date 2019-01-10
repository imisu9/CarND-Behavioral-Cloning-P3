'''
Read in input files
'''

import os
import csv

samples =[]
validation_ratio = 0.2

with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    samples.append(line)
    
from sklearn.model_selection import train_test_split

train_samples, validation_samples = train_test_split(samples, test_size=validation_ratio)

'''
Analyse the data, especially for distribution of angles
If needed, perform regularization and normalization on samples distribution
'''


'''
Define 'generator' function
'''

import cv2
import numpy as np
import sklearn

def binary_img(img, s_thresh=(170,255), sx_thresh=(20,100)):
  ############################
  # Image preprocessing
  # : similar to advanced line detection
  # : cropping could be done here
  # 1. cropping
  # 2. center (=mean to zero)
  # 3. normalization
  # 4. color space conversion to hls
  # 5. Image gradient, sobel x
  # 6. Random left-right flip
  ############################
  
  # cropping bottom 25px and top 65px 
  # examined pixs to exclude bonet area on the bottom and non-road area on the top
  cropped_img = img[25:95,:]
  # center (=mean to zero)
  # nomalization
  # convert to HLS color space and separate the L & S channel
  hls_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2HLS)
  l_channel = hls_img[:,:,1]
  s_channel = hls_img[:,:,2]
  # sobel x
  #   > take the derivative in x
  sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
  #   > absolute x derivatitve to accentuate lines away from horizontal
  abs_sobelx = np.absolute(sobelx)
  scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
  # threshold x gradient
  sxbinary = np.zeros_like(scaled_sobel)
  sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= s_thresh[1])] = 1
  # threshold color channel
  s_binary = np.zeros_like(s_channel)
  s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
  # > stack each channel to view their individual conttribution in green and blue respectively.
  # > this returns a stack of the two bianry images, whose components you can see as different colors.
  color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
  # combine the two binary thresholds
  combined_binary = np.zeros_like(sxbinary)
  combined_binary[(s_binary==1) | (sxbinary==1)] = 1
  
  return color_binary, combined_binary

batch_size = 32
correction = 0.25 # to be tuned

from sklearn.utils import shuffle

def generator(samples, batch_size=batch_size):
  num_samples = len(samples)
  while 1:  # Loop forever so the generator never terminates
    shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]
      
      images = []
      angles = []
      for batch_sample in batch_samples:
        path = '/opt/carnd_p3/data/IMG/'
        # center image
        center_image = cv2.imread(path + batch_sample[0].split('/')[-1])
        center_angle = float(batch_sample[3])
        center_color_bin, center_combined_bin = binary_img(center_image)
        images.append(center_combined_bin)
        angles.append(center_angle)
        # left image
        left_image = cv2.imread(path + batch_sample[1].split('/')[-1])
        left_angle = float(batch_sample[3]) - correction
        left_color_bin, left_combined_bin = binary_img(left_image)
        images.append(left_combined_bin)
        angles.append(left_angle)
        # right image
        right_image = cv2.imread(path + batch_sample[2].split('/')[-1])
        right_angle = float(batch_sample[3]) + correction
        right_color_bin, right_combined_bin = binary_img(right_image)
        images.append(right_combined_bin)
        angles.append(right_angle)
      
      # Trim image to only see section with road
      X_train = np.array(images)
      y_train = np.array(angles)
      yield shuffle(X_train, y_train)
      
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
row, col, ch = 70, 320, 3

model = Sequential()
#model.add(Cropping2D(cropping2D((65,25),(0,0)), input_shape=(ch, row, col), output_shape=(ch, row, col)))
# Convolutional layer: 5x5 kernel, 24@31x98
model.add(Conv2D(24, (5,5), strides=(2,2), padding='valid', input_shape=(row, col, ch)))
# Convolutional layer: 5x5 kernel, 36@14x47
model.add(Conv2D(36, (5,5), strides=(2,2), padding='valid'))
# Convolutional layer: 5x5 kernel, 48@5x22
model.add(Conv2D(48, (5,5), strides=(2,2), padding='valid'))
# Convolutional layer: 3x3 kernel, 64@3x30
model.add(Conv2D(64, (3,3), padding='valid'))
# Convolutional layer: 3x3 kernel, 64@1x18
model.add(Conv2D(64, (3,3), padding='valid'))
# Flatten layer: 1164 neurons
model.add(Flatten())
# Fully connected layer: 100 neurons
model.add(Dense(100, activation='tanh'))
# Fully connected layer: 50 neurons
model.add(Dense(50, activation='tanh'))
# Fully connected layer: 10 neurons
model.add(Dense(10, activation='tanh'))
# Fully connected output lyaer
model.add(Dense(1))

# Complie the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metircs=['accuracy'])

# Check the summary of this new model to confirm the architecture
from contextlib import redirect_stdout

with open('./examples/modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

from keras.utils import plot_model
plot_model(model, to_file='./examples/model.png')

from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint(filepath='model.h5', monitor='val_loss', save_best_only=True)
stopper = EarlyStopping(monitor='val_acc', min_delta=0.0003, patience=5)
history = model.fit_generator(train_generator,
                              steps_per_epoch=len(train_samples),
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
