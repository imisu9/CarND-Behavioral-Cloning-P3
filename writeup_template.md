# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[image8]: ./examples/nvidia_cnn-architecture.png "NVIDIA CNN Architecture"
[image9]: ./examples/accuracy.png "Accuracy"
[image10]: ./examples/loss.png "Loss"
[image11]: ./examples/final_train_angle_dist.png "Training data: Final Angle Distribution"
[image12]: ./examples/init_train_angle_dist.png "Training data: Initial Angle Distribution"
[image13]: ./examples/final_valid_angle_dist.png "Validation data: Final Angle Distribution"
[image14]: ./examples/init_valid_angle_dist.png "Validation data: Initial Angle Distribution"
[image15]: ./examples/model.png "Implemented Model"
[image16]: ./examples/original_angle_dist.png "Original Angle Distribution"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I have adopted NVIDIA CNN architecture described in [End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

![alt text][image8]

On a bare model, 
1. I added Activation function first. I started out with `tanh` then switched to `elu`.
2. Next, I've added lambda layer for normalization and mean subtraction.
3. Lastly, I've added BatchNormalization and Dropout layer.

The model summary:

![alt text][image15]

#### 2. Attempts to reduce overfitting in the model

Fighting overfit was tough part in terms of time and idea.
I've applied BatchNormalization and Dropout layers after every Activation layer.
The order of Activation, BatchNormalization, and Dropout layer was investigated,
and decided to stick with the order.
Next, I played with learning rate starting from `1e-4` to `1e-3`
Lastly, I've added L2 regularization to all Conv2D and FC layers.

#### 3. Model parameter tuning

The model used an adam optimizer with learning rate of `1e-4`.
However, once L2 regularization was introduced, loss did not decrease fast enough.
Loss got lowered to a proper level after learning rate increased upto `1e-3`

![alt text][image10]

#### 4. Appropriate training data

Preparing data was the hardest part. 
I used the given data at `/opt/carnd_p3/data/`.
The angle distribution of the original data look like this:

![alt text][image16]

The original distribution has very high bias on 0 angle.
I randomly chose one-eight of them to lessen the bias.

  if the steering==0, include only 1/8 of them.
  it will weaken exaggerated center-bias
  if np.array(line)[3] == ' 0':
    if np.random.randint(8) == 0:
      samples.append(line)
    else:
      samples.append(line)
            
![alt text][image12]

For augmentation,
1. Added Left-Right flipped image
2. Added Left and Right camera images
Now I have 6 times more images at any point of time with bell-shaped, standardized distribution.
The angle distribution of initial data look like this:

![alt text][image12]

At first, I thought preprocessing to binary image would help, which turned out to be false.
Convolution layer all took care of it at the end of the day.
