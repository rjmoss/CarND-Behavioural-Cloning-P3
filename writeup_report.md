# **Behavioral Cloning** 

## Robert Moss P3 writeup
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/original.jpg "Original"
[image2]: ./writeup/augmentations.jpg "Augmentations"
[image3]: ./writeup/original_distribution.jpg "Original Distribution"
[image4]: ./writeup/distribution.jpg "Improved Distribution"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode
My project includes the following files:
* clone.py containing the script to create and train the model (also clone_notebook.ipynb for a jupyter notebook version which was used during development).
* drive.py for driving the car in autonomous mode 
* model.h5 containing a trained convolution neural network 
* writeup_report.md (this file) summarizing the results
* run1.mp4 is a video of the model successfully driving the track

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable
    The clone.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. (Note that a python generator was not found to be necessary when training on an AWS GPU instance).

### Model Architecture and Training Strategy
#### 1. Creation of the Training Set & Training Process

The raw training data was a combination of **good driving**, **recovery driving** and **extra driving**. The **angle distribution** was balanced and **augmentation** was used to increase the size of the dataset and provide variety. Finally the images were cropped to remove unnecessary and distracting information (such as trees/sky).

##### Good driving
To capture good driving behavior, I recorded two laps on track one using center lane driving. On the first lap I went clockwise, and on the second lap I went anti-clockwise, this increased the amount of new training data (as the track looks different from each direction) and has the additional advantage of balancing the left/right bias of steering angles.

##### Recovery
During good driving the model only learns what to do when it is in/near the centre of the track, this means that if it leaves the centre it struggles to get back (unstable). I recoreded recovery clips with the vehicle starting at the left or right side of the track and recovering to the middle so that the model would learn how to act if/when it finds itself off centre. This reduced model drift as when the car stars to leave the centre it performs a recovery steering adjustment rather than continuing as if it's still in the centre.

##### Extra driving
Some additional clips were recorded in order to help the model on the less common and more difficult parts of the track. For example the model struggled with a corner which has just a dirt side, so I recorded more training data for that corner. In order to give the model the best chance of learning how to handle the corner I recorded a smooth/central corner and also some short recovery clips for offcentre positions while cornering.

##### Angle distribution
The dataset was quite biased towards the zero angle as a lot of the track involves little to no steering angle. In order to balance the dataset more effectively approximately half of the training data for steering angles with amagnitude less than 0.05 was thrown away. This lead to a more even distribution and improved the performance of the car on sharper corners.

Original distribution.
![][image3]

More balanced distribution.
![][image4]

(Note that the smaller peaks come from the centre peak +/- the correction for the left/right cameras - see augmentation).

##### Augmentation 

To augment the data sat, I also flipped images and angles as this would remove any left/right bias in the data set and generalise the model (e.g. flipped data from a left corner can help it learn how to steer a right corner).

After the collection process, I had 16836 number of data points. The images were then cropped to remove unnecessary/distracting information (such as the sky/trees) and also reduce the memory requirements and decrease training time.

The combination of lift/right cameras and flipping means that for one centre frame we get 6 training images/angles. The below images show the original image then the 6 variations (cropped):

![][image1]

![][image2]

The data was shuffled, with 20% being used as a validation set.

#### 2. Solution Design Approach

In designing the model architecture I first experimented with the LeNet architecture while working on the data processing (normalisation and cropping). This seemed to work reasonably well however the mean squared error was not decreasing much for either the training or validation set.

I then switched to the NVidia architecture.

When trying the trained model on the track the model struggled with a corner which has just a dirt side, so I recorded more training data for that corner. In order to give the model the best chance of learning how to handle the corner I recorded a smooth/central corner and also some short recovery clips for offcentre positions while cornering.

I found that adding more training data occasionally seemed to make the model worse, with the car unable to steer successfully around areas of the track it previously managed. I therefore had to be very careful when adding more data, and found sometimes that removing excess data improved the model.

#### 3. Final Model Architecture
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

The model includes RELU layers to introduce nonlinearity, and the data is cropped then normalized in the model using 2 Keras lambda layers.

I used the adam optimizer so there was no need to manually train the learning rate.

It was observed that training beyond 1-3 epochs decreased the training loss but not the validation loss (and the performance on the track decreased) implying overfitting so the training was generally stopped after only 1-3 epochs. The final model was only trained for 1 epoch.

### Simulation
See video run1.mp4 for a video of the model successfully completing a lap of the simulation.

