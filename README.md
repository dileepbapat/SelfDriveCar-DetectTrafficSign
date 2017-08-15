#**Traffic Sign Recognition** 

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report
* Visualize activation at different layers.


[//]: # (Image References)

[image1]: images/train_spread.png "Train set images by sign"
[image2]: images/valid_spread.png "Validation set images by sign"
[image3]: images/test_spread.png "Test set images by sign"
[image4]: images/grayscale.png "Gray scale trffic sign image"
[image5]: images/sample_color.png "Traffic sign - Original image"
[newimage1]: ./data/1.jpg "Traffic Sign 1"
[newimage2]: ./data/2.jpg "Traffic Sign 2"
[newimage3]: ./data/3.jpg "Traffic Sign 3"
[newimage4]: ./data/4.jpg "Traffic Sign 4"
[newimage5]: ./data/5.jpg "Traffic Sign 5"
[prediction]: images/prediction.png "Prediction"
[features]: images/features.png "Features in hidden layers"
[probability_distribution]: images/probability_distribution.png "Probability of Children crossing sign distribution"



---
###Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/dileepbapat/SelfDriveCar-DetectTrafficSign/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Exploratory visualization of the dataset.

First, lets see whats the spread of dataset by target labels i.e., traffic sign.

![Traffic sign type spread on training data][image1]

If we compare same on validation and test:

![Traffic sign type spread on validation data][image2]

by looking at above graphs we can confirm that train and validation set split is not skewed. Both are having similar distribution of number of samples per sign.
However the dataset is skewed in number of samples per sign, looks like about 70% of images represent 30% of sign.

Just to confirm test set is also similar we can plot similar graph for test.

![Traffic sign type spread on test data][image3]

###Design and Test a Model Architecture

####1. Pre-processing
As a first step, I decided to convert the images to gray-scale because shapes in images are enough to distinguish the sign, color is not adding much to it. 

Here is an example of a traffic sign image before and after gray-scaling.

![Original image][image5] 
![Grayscale image][image4] 

As a last step, I normalized the image data because keeping the input in range of -1 to 1 helps in avoiding numerical computation issues around exponential growth of weights. 

For now I am not augmenting data, after verification of results if its still low, I will generate more images to balance the class count in training images.
     

####2. Network

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray scale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6    				|
| Fully connected		| 400 to 120   									|
| RELU					|												|
| dropout				| .6											|
| Fully connected		| 120 to 84   									|
| RELU					|												|
| dropout				| .6											|
| Fully connected		| 84 to 21   									|
| RELU					|												|
| dropout				| .6											|
| Fully connected		| 21 to 43   									|
|						|												|
|						|												|
 


####3. Training

* I started with basic LeNet model with default parameters suggested, learning rate of 0.001 and batch size of 128. 
* As accuracy needed was more than 93% and LeNet was not getting accuracy little below it, I added an additional layer
* Drop out of .4 initial and then reduced it to .3 in training process.
* Training was done with following phases.
    * 15 rounds with .4 drop out accuracy range of .41 to .88
    * 5 rounds with .4 again yielded accuracy .87 .89
    * multiple rounds of 25 epochs training brought accuracy of .95
    * final round of 25 epoch with drop out .3 got accuracy ~ .96

* Things to try on this network 
    * experiment with learning rate,
    * change / add layers.
   
####4. Iterative development

My final model results were:
* training set accuracy of 99.95%
* validation set accuracy of 93.5 
* test set accuracy of 95.5

If an iterative approach was chosen:
* Started with LeNet architecture and no drop outs.
* LeNet architecture was getting accuracy close to .9 
* Added a fully connected layer and dropout layers at 2 points in the network.
* Used default learning rate 0.001 however number of epoch the training was done increased to many folds.
* As input is image and 2 dimensional in nature, each pixel will have relation with adjacent pixels so 2d convolutional 
network will help in identifying shapes in first phase. To get more higher level shapes as features a maxpool and again conv2d 
was used. To control the overfitting to sample dropout layer was used.

If a well known architecture was chosen:
* Initial architecture was just LeNet architecture.
* As traffic sign recognition includes image/shape identification a simple way to get started is to use existing network 
that works for similar problem area. so LeNet was used as bootstraping.
* Training accuracy reached close to 99% is evidence that network is good enough to fit the problem however its suffering 
from overfitting as there are not many images in some of classes. adding regularization helped in controlling validation 
 accuracy.
 

###Testing the model on new images

####1. Download images from internet

Here are five German traffic signs that I found on the web: (it was resized to 32x32 to match the dataset)

![alt text][newimage1] ![alt text][newimage2] ![alt text][newimage3] 
![alt text][newimage4] ![alt text][newimage5]


####2. Prediction on new images

Here are the results of the prediction:

| Image			                |     Prediction      					| 
|:-----------------------------:|:-------------------------------------:| 
| Turn right ahead      		| Turn right ahead   					| 
| Road work     			    | Road work 							|
| Bumpy road					| Bumpy road							|
| Children crossing	      		| Children crossing						|
| No entry 			            | No entry      						|

Below is picture of fresh new picture downloaded from internet and predicted class images from train set. Image on 
left side is gray scale image (drawn without cmap=gray so it looks different color) and right 8 images are from 
train set.

![Prediction on new set][prediction] 

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.5%
(as given dataset is only 5 the resolution of difference is 20%)

####3. Prediction confidence

The code for making predictions on my final model is located below #Prediction-confidence heading.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| Turn right ahead   							| 
| 1.00     				| Road work  									|
| 0.99					| Bumpy road									|
| 0.88	      			| Children crossing	    		 				|
| 1.0				    | No entry            							|


For the fourth image Children crossing probability is lesser compared to others. One thing that separates from train set is its 
skewed a bit. Possibly if we can add augmented data with some rotation it should help.

Lets visualize the probability distribution of this sample (Children crossing):

Other closest match are General caution and Pedestrians

![Probability of Children crossing sign][probability_distribution]

### Visualizing internal activation at hidden layers
####1. Feature map at first convolution layer
Below is activation values visualized as image when new image data was fed.

![Hidden layer visualization][features] 
