# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 18:08:00 2020

@author: Mukul
"""

# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__ #prints version of tensor flow

# Part 1 - Data Preprocessing

# Generating images for the Training set
# ImageDataGenerator 
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True) #'Shear' means that the image will be distorted along an axis, mostly to create or rectify the perception angles

# Generating images for the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)

# Creating the Training set
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary') #class mode depicts result type

# Creating the Test set
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[64, 64, 3]))
#2D convolutional layer function. 32 layered filter, linear rectification activation function, input image size 64 and 3 denotes color image
 
# Step 2 - Pooling
#Pooling reduces size and maintains features
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid')) 
#Pool size is 2x2, stride is the number of pixels the window is moved, padding is when the window exceeds the frame & is replaced by 0s

# Adding a second convolutional layer, remove input shape as it's reqd only for the first layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid')) 

# Step 3 - Flattening - converts into a single column
cnn.add(tf.keras.layers.Flatten())                                              

# Step 4 - Full Connection - units is the number of nuerons
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer - units is 1 because it's binary classification, for op layer we use sigmoid function
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# part 4 - Training the CNN on the Training set and evaluating it on the Test set
cnn.fit (x =  training_set, validation_data = test_set, epochs = 25) #After running through each epoch, the accuracy increases
# In this case accuracy of prediction on training set was 0.89 and 0.80 for the test set


# Part 5 - Making a single prediction
import numpy as np
from keras.preprocessing import image
test_img = image.load_img('/root/.jpg', target_size = (64,64))
test_img = np.expand_dims(test_img, axis = 0) #image expanded in first dimension, axis = 0
result = cnn.predict(test_img)
#Call class indices attribute from training set object
training_set.class_indices
#result is in a batch- access first image of the first batch
if result [0][0] == 1:
    prediction = 'Dog' #Predicts a dog image if the result image is a dog
else:
    prediction = 'Cat' #Predicts a cat image if the result image is a cat
    
# Prnt the result
print (prediction)



