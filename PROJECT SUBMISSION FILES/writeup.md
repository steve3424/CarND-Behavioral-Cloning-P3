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
-3 convolution layers sized 5x5 with a stride of 2
-2 convolution layers sized 3x3 with a stride of 1
-RELU activations after each convolution layer
-dropout with a 0.5 probability after each convolution layer
-3 fully connected layers
-1 output layer
-loss function = mean squared error
-optimizer = adam optimizer

#### 2. Attempts to reduce overfitting in the model

I used dropout with a keep probability of 0.5 after each convolution. I experimented with l2 regularization on the fully connected layers, but found this did not improve my results.

The validation loss was generally lower than training loss and both decreased monotonically over 3 epochs.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

I kept the train/valditation split at 80/20 and trained over 3 epochs.

#### 4. Appropriate training data

I collected a variety of training data including multiple laps of center lane driving, center lane driving in the reverse direction, recovery data from left and right sides of the road, and extra turning data to prevent a bias toward driving straight.

My simulation ended up driving successfully around the track by training only on 2 forward and 2 backward laps. 



## Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with an architecture that I thought could work and tweak based on the results of the simulation run.

I decided to stick with the NVIDIA PilotNet model as a starting point. Not only was this suggested in the lectures, but NVIDIA's paper (https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) showed that with the right training data this model is successful in real world testing.

There were 2 things I was looking at to tweak the model to be successful:
1. training loss vs validation loss (overfitting)
2. how well the model was driving in the simulation

Using the model as it was, there was a problem with overfitting. My validation loss was consintently higher than my training loss and occasionally the loss would increase over the course of training. Adding dropout after each convolution and decreasing the epochs from 5 to 3 solved this problem.

I settled on this model and decided then to see what data I needed to train on in order to complete a successful lap. I started with one lap of center lane driving data. This didn't work so I added a second lap of center lane driving data. This was a little bit better, but still had some problems with a left turn bias even during straightaways. The car would slowly drift off the road to the left and never return to center.

I then decided to add a lap of center lane driving in the reverse direction. This improved the model, but it still had trouble with the final right turn of the track so I added a second lap in the reverse direction. These 2 laps helped with the left turn bias and allowed it to successfully complete a full lap of the track.

Using only 4 laps of center lane driving, 2 in each direction, did not result in the cleanest lap around the track, but for the purposes of this project it was successful.

#### 2. Possible Improvements

The primary area I would like experiment with is the training data. I would like to try out a more robust collection of training data (extra turning data, recovery data, different tracks) to see if the lap could be cleaner and to see if the model could generalize to other tracks.
