"""
	Author:    Prahar Bhatt
	Created:   10.05.2020
	Center for Advanced Manufacturing, University of Southern California.
"""

# Importing libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
# %matplotlib inline 
import matplotlib.pyplot as plt
import math

# Function to load data
def MINST():

  # Downloading MISNT image data
  (xTrain, yTrain),(xTest, yTest) = \
  tf.keras.datasets.mnist.load_data(path="mnist.npz")

  # Normalizing training data
  xTrainNorm = []
  for i in range(0,len(xTrain)):
    xTrainNorm.append((xTrain[i] / 255))
  
  # Normalizing test data
  xTestNorm = []
  for i in range(0,len(xTest)):
    xTestNorm.append((xTest[i] / 255))

  # Returning the test and train data
  return [np.asarray(xTrainNorm), yTrain, np.asarray(xTestNorm), yTest]

# Function to generate the network
def CNN():
    
    # Initilizing the model
    model = tf.keras.Sequential()

    # Additing convolutional layer
    model.add(keras.layers.Conv2D(16, kernel_size=(3, 3),padding="same", 
                                  activation='relu', input_shape=(28, 28, 1)))
    # Additing max pooling layer
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Additing convolutional layer
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),padding="same", 
                                  activation='relu'))
    # Additing max pooling layer
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Additing convolutional layer
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),padding="same", 
                                  activation='relu'))
    # Additing max pooling layer
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Additing convolutional layer
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3),padding="valid", 
                                  activation='relu'))

    # Adding flattening layer
    model.add(keras.layers.Flatten())

    # Adding fully connected layer
    model.add(keras.layers.Dense(32, activation='relu'))

    # Adding softmax output layer
    model.add(keras.layers.Dense(16, activation='softmax'))
            
    # Defining how to complie the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.1),
                  loss='sparse_categorical_crossentropy',
                  metrics = ['accuracy'])

    # Returning the model
    return model

# Execution begins here
if __name__ == "__main__":

  # Obtaining MINST data
  trainNX, trainY, testNX, testY = MINST()

  # Defining plotting variables
  plt.rcParams['figure.figsize'] = 10, 10
  Image_figure, Image_axis = plt.subplots(5, 5)
  Image_figure.suptitle("25 Training images", size = 16)
  Image_figure.tight_layout(pad=3.0)

  # Plotting 5 x 5 training images
  for i in range(0,5):
    for j in range(0,5):
      idI = i*5 + j
      Image_axis[i,j].imshow(trainNX[idI])
      Image_axis[i,j].set_title("Id: "+ str((idI+1)))

  # Reshaping image data
  trainNX = trainNX.reshape(60000,28,28,1)
  testNX = testNX.reshape(10000,28,28,1)

  # Creating CNN
  modelCNN = CNN()

  # Training CNN
  histories = modelCNN.fit(trainNX, trainY, batch_size=128, epochs=3, verbose=1, 
               validation_data=(testNX, testY), 
               workers=4, use_multiprocessing=True)

# Plotting accuracy of training and test data
  plt.rcParams['figure.figsize'] = 10, 5
  Accuracy_figure, Accuracy_axis = plt.subplots(1, 2)
  Accuracy_figure.suptitle("Epochs vs Accuracy", size = 16)
  Accuracy_figure.tight_layout(pad=3.0)
  Accuracy_axis[0].plot(histories.history['accuracy'])
  Accuracy_axis[0].legend(("training data",))
  Accuracy_axis[1].plot(histories.history['val_accuracy'])
  Accuracy_axis[1].legend(("testing data",))

# Obtaining the the first convolutional layer
  layer = modelCNN.layers[0]
  
  # Getting the filters pf the first convolutional layer
  filters, biases = layer.get_weights()

  # Defining the plot for the filers
  plt.rcParams['figure.figsize'] = 20, 10
  Filter_figure, Filter_axis = plt.subplots(4, 8)
  Filter_figure.suptitle("Filters", size = 16)
  Filter_figure.tight_layout(pad=3.0)

  # Plotting the 32 (3x3) filters in 4 x 8 grid with colorbar
  for i in range(0,4):
    for j in range(0,8):
      idF = i*8 + j

      # Subtracting respective means
      meanFilters = filters[:,:,0,idF] - np.mean(filters[:,:,0,idF])
      
      # Generate colorbar limits for zero center
      fmax = meanFilters.max()
      fmin = meanFilters.min()
      if abs(fmax) > abs(fmin):
        fmin = - abs(fmax)
      else:
        fmax = abs(fmin)

      # Plot filters with colorbar
      plt.colorbar(Filter_axis[i,j].imshow(meanFilters, 
                                           vmin = fmin, vmax = fmax),
                   ax=Filter_axis[i,j], orientation='horizontal')
      Filter_axis[i,j].set_title("Id: "+ str((idF+1)))