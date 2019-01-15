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
import numpy as np

original_samples = []
samples = []

with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        original_samples.append(line)
        
        # if the steering==0, include only 1/8 of them.
        # it will weaken exaggerated center-bias
        if np.array(line)[3] == ' 0':
            if np.random.randint(8) == 0:
                samples.append(line)
        else:
            samples.append(line)
        
    # delete the first line since it contains headers like 'center', 'left', etc.
    original_samples = original_samples[1:]
    samples = samples[1:]
    #print(len(samples))
    
from sklearn.model_selection import train_test_split

validation_ratio = 0.2
train_samples, validation_samples = train_test_split(samples, test_size=validation_ratio)

# Check angle distribution on train_samples and validation_samples
angle_distribution(original_samples, './examples/original_angle_dist.png')
angle_distribution(train_samples, './examples/init_train_angle_dist.png')
angle_distribution(validation_samples, './examples/init_valid_angle_dist.png')
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

  #return np.expand_dims(color_binary, axis=2), np.expand_dims(combined_binary, axis=2)
  return np.expand_dims(combined_binary, axis=2)

'''
def load_data(batch_sample):
    path = '/opt/carnd_p3/data/IMG/'
    correction = 0.2 # to be tuned
    images = []
    angles = []
    
    # load data
    # center image
    image = cv2.imread(path + batch_sample[0].split('/')[-1])
    angle = float(batch_sample[3])
    images.append(image)
    angles.append(angle)
    
    # center image flipped
    images.append(np.fliplr(image))
    angles.append(-angle)
    # left image
    image = cv2.imread(path + batch_sample[1].split('/')[-1])
    angle = float(batch_sample[3]) + correction
    images.append(image)
    angles.append(angle)
    # left image flipped
    images.append(np.fliplr(image))
    angles.append(-angle)
    # right image
    image = cv2.imread(path + batch_sample[2].split('/')[-1])
    angle = float(batch_sample[3]) - correction
    images.append(image)
    angles.append(angle)
    # right image flipped
    images.append(np.fliplr(image))
    angles.append(-angle)
    
    # cropping bottom 25px and top 65px
    # examined pixs to exclude bonet area on the bottom and non-road area on the top
    #cropped_img = img[25:95,:]
    # mean subtraction
    #mean_sub = rand_flip - np.mean(rand_flip)
    # normalize after mean substraction
    #norm = mean_sub / np.std(mean_sub)
    # call binary_img()
    # > color space conversion to hls
    # > image gradient, sobelx
    #color_bin, combined_bin = binary_img(norm)
        
    return images, angles
'''
batch_size = 32

from sklearn.utils import shuffle

def generator(samples, batch_size=batch_size):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            path = '/opt/carnd_p3/data/IMG/'
            correction = 0.2 # to be tuned
            images = []
            angles = []
            
            for batch_sample in batch_samples:
                # center image
                image = cv2.imread(path + batch_sample[0].split('/')[-1])
                #image = binary_img(image)
                angle = float(batch_sample[3])
                images.append(image)
                angles.append(angle)
                
                # center image flipped
                images.append(np.fliplr(image))
                angles.append(-angle)
                # left image
                image = cv2.imread(path + batch_sample[1].split('/')[-1])
                #image = binary_img(image)
                angle = float(batch_sample[3]) + correction
                images.append(image)
                angles.append(angle)
                # left image flipped
                images.append(np.fliplr(image))
                angles.append(-angle)
                # right image
                image = cv2.imread(path + batch_sample[2].split('/')[-1])
                #image = binary_img(image)
                angle = float(batch_sample[3]) - correction
                images.append(image)
                angles.append(angle)
                # right image flipped
                images.append(np.fliplr(image))
                angles.append(-angle)
                
            X_train = np.array(images)
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

from keras.layers import Input, Lambda, Cropping2D, BatchNormalization
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras import optimizers, regularizers
import tensorflow as tf

# Trimmed image format
row, col, ch = 160, 320, 3

model = Sequential()
# Preprocess imcoming data, centered around zero with small standard deviation
# mean subtraction and normalization
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=(row, col, ch)))
# Cropping top 70 pix and bottom 25 pix
model.add(Cropping2D(cropping=((70,25), (0,0))))
# Convolutional layer: 5x5 kernel, 24@31x98
#model.add(Conv2D(24, (5,5), strides=(2,2), padding='valid', input_shape=(row, col, ch)))
model.add(Conv2D(24, (5,5), strides=(2,2), padding='valid', kernel_regularizer=regularizers.l2(0.001)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(.5))
# Convolutional layer: 5x5 kernel, 36@14x47
model.add(Conv2D(36, (5,5), strides=(2,2), padding='valid', kernel_regularizer=regularizers.l2(0.001)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(.5))
# Convolutional layer: 5x5 kernel, 48@5x22
model.add(Conv2D(48, (5,5), strides=(2,2), padding='valid', kernel_regularizer=regularizers.l2(0.001)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(.5))
# Convolutional layer: 3x3 kernel, 64@3x30
model.add(Conv2D(64, (3,3), padding='valid', kernel_regularizer=regularizers.l2(0.001)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(.5))
# Convolutional layer: 3x3 kernel, 64@1x18
model.add(Conv2D(64, (3,3), padding='valid', kernel_regularizer=regularizers.l2(0.001)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(.5))
# Flatten layer: 1164 neurons
model.add(Flatten())
# Fully connected layer: 100 neurons
model.add(Dense(100, kernel_regularizer=regularizers.l2(0.001)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(.5))

# Fully connected layer: 50 neurons
model.add(Dense(50, kernel_regularizer=regularizers.l2(0.001)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(.5))

# Fully connected layer: 10 neurons
model.add(Dense(10, kernel_regularizer=regularizers.l2(0.001)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(.5))
# Fully connected output lyaer
model.add(Dense(1, kernel_regularizer=regularizers.l2(0.001)))

# Complie the model
adam = optimizers.Adam(lr=1e-3)
model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])

# Check the summary of this new model to confirm the architecture
from contextlib import redirect_stdout

with open('./examples/modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

from keras.utils import plot_model
plot_model(model, to_file='./examples/model.png')

from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint(filepath='model.h5', monitor='val_loss', save_best_only=True)
stopper = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience=3)
# steps_per_epoch and validation_step are multiplied by 6 
# since generator added center/left/right and their flipped data on the fly
history = model.fit_generator(train_generator,
                              steps_per_epoch=int(len(train_samples)*6/batch_size),
                              validation_data=validation_generator,
                              validation_steps=int(len(validation_samples)*6/batch_size),
                              epochs=15,
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
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('./examples/loss.png')
#plt.show()
plt.cla()

model.save('model.h5')