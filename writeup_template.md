# **Behavioral Cloning** 

## Robert Moss P3 submission
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

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md (this file) summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

In designing the model architecture I first experimented with the LeNet architecture while working on the data processing (normalisation and cropping). This seemed to work reasonably well however the mean squared error was not decreasing much for either the 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

When trying the trained model on the track the model struggled with a corner which has just a dirt side, so I recorded more training data for that corner. In order to give the model the best chance of learning how to handle the corner I recorded a smooth/central corner and also some short recovery clips for offcentre positions while cornering.

I found that adding more training data occasionally seemed to make the model worse, with the car unable to steer successfully around areas of the track it previously managed. I therefore had to be very careful when adding more data, and found sometimes that removing excess data improved the model.

#### 2. Final Model Architecture
I experimented with the LeNet architecture while working on the data processing but  for the final model I am using the  nvidia architecture. Including the data preprocessing, this consists of the following layers:


| Layer         		|     Description	        					| 
|:----------------------|:----------------------------------------------| 
| Input         		| 160x320x3 RGB image   						|
| Crop         			| 100x320x3 RGB image   						|
| Normalise         	| 100x320x3 									|
| Convolution 5x5    	| 24 filters, 2x2 stride, RELU activation 		|
| Convolution 5x5    	| 36 filters, 2x2 stride, RELU activation 		|
| Convolution 5x5    	| 48 filters, 2x2 stride, RELU activation 		|
| Convolution 3x3    	| 64 filters, RELU activation 					|
| Convolution 3x3    	| 64 filters, RELU activation 					|
| Flatten				| output size 1164								|
| Fully connected 100	| output size 50								|
| Fully connected 50 	| output size 10								|
| Fully connected 10 	| output size 1									|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had X number of data points. I then preprocessed this data by ...

The data was shuffled, with 20% being used as a validation set.
I used the adam optimizer so there was no need to manually train the learning rate.
