# **Behavioral Cloning Writeup** 


#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

PROJECT FILES:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* run1.mp4 the recording of my lap around the track 
* writeup.md



## Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The starting point for my model was NVIDIA's PilotNet as suggested in the classroom lectures. This is a convolutional neural network with the following pipeline:

-pre-processing layer to normalize image values -0.5 to 0.5
-pre-processing layer to crop the background off of the top of the image and the front bumper off of the bottom
-3 convolution layers sized 5x5 with a stride of 2 with RELU activations
-2 convolution layers sized 3x3 with a stride of 1 with RELU activations
-3 fully connected layers with RELU activations
-1 drop out regularization after the first FC layer
-1 output layer
-loss function = mean squared error
-optimizer = adam optimizer

Below is a visualization of the network architecture

![Alt text](data_images/network.png?raw=True "Network")

#### 2. Attempts to reduce overfitting in the model

I used dropout with a keep probability of 0.5 after each convolution. I experimented with l2 regularization on the fully connected layers, but found this did not improve my results.

The validation loss was generally lower than training loss and both decreased monotonically over 3 epochs.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

I kept the train/valditation split at 80/20 and trained over 3 epochs.

#### 4. Appropriate training data

I collected a variety of training data including multiple laps of center lane driving, center lane driving in the reverse direction, recovery data from left and right sides of the road, and extra turning data to prevent a bias toward driving straight.

My simulation ended up driving successfully by using 5 forward laps, 3 backward laps, and extra turning data.

![Alt text](data_images/center_2018_07_23_21_16_44_095.jpg?raw=True "Center")
![Alt text](data_images/left_2018_07_23_21_16_44_095.jpg?raw=True "Left")
![Alt text](data_images/right_2018_07_23_21_16_44_095.jpg?raw=True "Right")

Above are an example of the center, left, and right images I used for training. I added the left and right images with a +0.2 and -0.2 steering correction respectively


## Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with an architecture that I thought could work and tweak based on the validation loss and the results of the simulation run.

I decided to stick with the NVIDIA PilotNet model as a starting point. Not only was this suggested in the lectures, but NVIDIA's paper (https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) showed that with the right training data this model is successful in real world testing.

There were 2 things I was looking at to tweak the model to be successful:
1. training loss vs validation loss (overfitting)
2. how well the model was driving in the simulation

I recorded 5 laps in either direction on the first track as well as extra data for each turn and some recovery data. I decided to train on one lap and continually add data to see if I could find success that way.

I found that with multiple forward and backward laps with slightly more forward lap data compared to backward lap data was a successful strategy.

Using the model as it was, there was a problem with overfitting. My validation loss would quickly hit its minimum value and then begin to increase over the course of training. Adding dropout after the first fully connected layer and keeping my epochs to no more than 5 solved this problem. More regularization could have potentially improved the model, but it wasn't necessary to complete a lap.



#### 2. Possible Improvements

The primary area I would like experiment with is the training data. I would like to try out a more robust collection of training data (extra turning data, recovery data, different tracks) to see if the lap could be cleaner and to see if the model could generalize to other tracks.
