'''
Distribution function for angles,
to analyse the data, especially for distribution of angles.
If needed, perform regularization and normalization on samples distribution
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def angle_distribution(angles, file_path):
    
    num_bins = 50
    if type(angles) is list:
        x = np.array(angles)
    else:
        x = angles
    if x.ndim != 1:
        x = x[:, 3]
    x = x.astype(np.float)
    mu = np.mean(x)
    sigma = np.std(x)
    
    fig, ax = plt.subplots()
    
    # the histogram of the data
    n, bins, patches = ax.hist(x, num_bins, normed=1)
    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    ax.plot(bins, y, '--')
    ax.set_xlabel('Angles')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Steering Angle')
    
    # tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.savefig(file_path)
    #plt.show()
    plt.cla()

'''
Read in input files
'''

import os
import csv

samples =[]

with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    samples.append(line)
  # delete the first line since it contains headers like 'center', 'left', etc.
  samples = samples[1:]
    
from sklearn.model_selection import train_test_split

validation_ratio = 0.2
train_samples, validation_samples = train_test_split(samples, test_size=validation_ratio)

# Check angle distribution on train_samples and validation_samples
angle_distribution(train_samples, './examples/init_train_angle_dist.png')
angle_distribution(validation_samples, './examples/init_valid_angle_dist.png')
'''
# Scale data: zero mean and unit variance
from sklearn import preprocessing
import numpy as np
train_samples = np.array(train_samples)
train_samples[:,3] = preprocessing.scale(train_samples[:,3])
train_samples = train_samples.tolist()
validation_samples = np.array(validation_samples)
validation_samples[:,3] = preprocessing.scale(validation_samples[:,3])
validation_samples = validation_samples.tolist()
# Check angle distribution on train_samples and validation_samples
angle_distribution(train_samples, './examples/scaled_train_angle_dist.png')
angle_distribution(validation_samples, './examples/scaled_valid_angle_dist.png')
'''
'''
Define 'generator' function
'''

import cv2
import numpy as np
import sklearn

def binary_img(img, s_thresh=(170,255), sx_thresh=(20,100)):
  
  # convert to HLS color space and separate the L & S channel
  hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
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
  
  return np.expand_dims(color_binary, axis=2), np.expand_dims(combined_binary, axis=2)

def preprocessing(path, angle):
  ############################
  # Image preprocessing
  # : similar to advanced line detection
  # : cropping could be done here
  # 1. cropping
  # 2. color space conversion to hls
  # 3. image gradient, sobel x
  ############################
  
  # load data
  img = cv2.imread(path)
  # cropping bottom 25px and top 65px 
  # examined pixs to exclude bonet area on the bottom and non-road area on the top
  cropped_img = img[25:95,:]
  # call binary_img()
  # > color space conversion to hls
  # > image gradient, sobelx
  color_bin, combined_bin = binary_img(cropped_img)
  # randomly flip left-right
  if np.random.randint(2) == 0:
    combined_bin = np.fliplr(combined_bin)
    angle = -angle
  
  return combined_bin, angle

batch_size = 32
correction = 0.2 # to be tuned

from sklearn.utils import shuffle

def generator(samples, batch_size=batch_size):
  num_samples = len(samples)
  while 1:  # Loop forever so the generator never terminates
    shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]
      
      images = []
      angles = []
      path = '/opt/carnd_p3/data/IMG/'
      
      for batch_sample in batch_samples:
        # take random choice among center, left and right image
        # 0 = center, 1 = left, 2 = right
        choice = np.random.randint(3)
        if choice == 0:
            # center image
            image, angle = preprocessing(path + batch_sample[0].split('/')[-1], float(batch_sample[3]))
        elif choice == 1:
            # left image
            image, angle = preprocessing(path + batch_sample[1].split('/')[-1], float(batch_sample[3])+correction)
        elif choice == 2:
            # right image
            image, angle = preprocessing(path + batch_sample[2].split('/')[-1], float(batch_sample[3])-correction)
        images.append(image)
        angles.append(angle)
        
      X_train = np.array(images)
      # Check angle distribution on train_samples and validation_samples
      #angle_distribution(angles, './examples/batch_angle_dist.png')
      y_train = np.array(angles)      
      yield shuffle(X_train, y_train)
      
# Complie and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Take batch_size = 2048 to analyze data distribution
train_X_gen, train_y_gen = [], []
val_X_gen, val_y_gen = [], []

train_X_gen, train_y_gen = next(generator(train_samples, batch_size=2048))
val_X_gen, val_y_gen = next(generator(validation_samples, batch_size=2048))

angle_distribution(train_y_gen, './examples/final_train_angle_dist.png')
angle_distribution(val_y_gen, './examples/final_valid_angle_dist.png')

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
row, col, ch = 70, 320, 1

model = Sequential()
# Preprocess imcoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=(row, col, ch), output_shape=(row, col, ch)))
# Convolutional layer: 5x5 kernel, 24@31x98
#model.add(Conv2D(24, (5,5), strides=(2,2), padding='valid', input_shape=(row, col, ch)))
model.add(Conv2D(24, (5,5), strides=(2,2), padding='valid'))
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
model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])

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
plt.subplots()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('./examples/accuracy.png')
#plt.show()
plt.cla()

# Plot training & validation loss values
plt.subplots()
plt.plot(history.history['loss']
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('./examples/loss.png')
#plt.show()
plt.cla()

model.save('model.h5')
