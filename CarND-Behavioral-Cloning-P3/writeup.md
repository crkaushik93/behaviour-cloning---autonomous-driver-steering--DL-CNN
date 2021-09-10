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

[image1]: ./examples/image1.png "Model Architecture"
[image2]: ./examples/image2.png "Output Image"
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
The model provided by the Nvidia is tested here. For the model architecture as an input we considered image of the shape (160,320,3).We decided to keep the architecture  same for the remaining model but feed an image of different input shape.

### Loading The Data:
We considered the the dataset provided by Udacity
Used the  OpenCV libraries to load the images. The loaded images are read as BGR by default so it is converted to RGB format.
Since the steering angle is associated with the images, a correction factor is used  for both left and right images as the steering angle is calculated  based on the center angle.
We considered a correction factor of 0.2
For the left images  the steering angle is increased by 0.2 and for the right images decrease by 0.2.

### Preprocessing:
The concept of  shuffle the images is introduced here to change the order in which images comes such that it doesn't  matters to the CNN.
Augmenting the data- we decided to flip the image horizontally and adjust steering angle accordingly, we used cv2 to flip the images.
During augmenting after flipping we multiplied the steering angle by a factor of -1 to get the steering angle for the flipped image.
Through this approach we were able to generate 6 images for  one entry in .csv file

### Creating Training and Validation set:
We split the dataset into training and validation set using sklearn preprocessing library.
We kept 15% of the data in Validation Set and remaining in Training Set.

#### 1.Final Model Architecture:
We made  a little changes to the original architecture, so our final architecture looks like in the image below.
![alt text][image1]

As from the model summary my first step is to apply normalization to the all the images.
Second step is to crop the image 70 pixels from top and 25 pixels from bottom. 
Next Step is to define the first convolutional layer with filter depth as 24 and filter size as (5,5) with (2,2) stride followed by ELU activation function
The second convolutional layer with filter depth as 36 and filter size as (5,5) with (2,2) stride followed by ELU activation function
The third convolutional layer with filter depth as 48 and filter size as (5,5) with (2,2) stride followed by ELU activation function
Next we defined two convolutional layer with filter depth as 64 and filter size as (3,3) and (1,1) stride followed by ELU activation funciton
Next step is to flatten the output from 2D to side by side
Here we applied first fully connected layer with 100 outputs
Here we introduce Dropout with Dropout rate as 0.25 to combact overfitting
Next we introduced second fully connected layer with 50 outputs
Then  a third connected layer with 10 outputs
And finally the layer with one output.

#### 2. Attempts to reduce overfitting in the model

After the full connected layer we used a dropout so that the model generalizes on a track that it has not seen. we decided to keep the Dropoout rate as 0.25 to combact overfitting.

#### 3. Model parameter tuning

No of epochs= 5
Optimizer Used- Adam
Learning Rate- Default 0.001
Validation Data split- 0.15
Generator batch size= 32
Correction factor- 0.2
Loss Function Used- MSE(Mean Squared Error).

### Output Video:
Here is the image from the output video.
![alt text][image2]