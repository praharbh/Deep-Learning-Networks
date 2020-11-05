"""
	Author:    Prahar Bhatt
	Created:   09.21.2020
	Center for Advanced Manufacturing, University of Southern California.
"""

# Importing libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
# %matplotlib inline 
import matplotlib.pyplot as plt
import math

# Function to generate the network
def MLP(Input_Dim=1,Output_Dim=1,Width=10,Depth=15,Reg_Param=0.0):

    # Checking of depth is greater than the threshold
    assert Depth > 1, 'Depth of generator must be greater than 1'
    
    # Initilizing the model
    model = tf.keras.Sequential()
    
    # Adding first hidden layer
    model.add(keras.layers.Dense(Width, input_shape=(Input_Dim,), 
    	activation='sinh', use_bias=True, 
    	kernel_initializer=keras.initializers.RandomUniform(0,1), 
    	bias_initializer=keras.initializers.RandomUniform(0,1), 
    	kernel_regularizer=keras.regularizers.l2(Reg_Param)))

    # Adding remaining hidden layers
    if(Depth > 2):
        for l in range(Depth - 2):
            model.add(keras.layers.Dense(Width, activation='sinh',use_bias=True, 
            	kernel_initializer=keras.initializers.RandomUniform(0,1), 
            	bias_initializer=keras.initializers.RandomUniform(0,1),
            	kernel_regularizer=keras.regularizers.l2(Reg_Param)))
            
    # Adding output layer
    model.add(keras.layers.Dense(Output_Dim, activation=None, use_bias=True, 
    	kernel_initializer=keras.initializers.RandomUniform(0,1), 
    	bias_initializer=keras.initializers.RandomUniform(0,1), 
    	kernel_regularizer=keras.regularizers.l2(Reg_Param)))
    
    # Defining how to complie the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-2),
                  loss='mse',
                  metrics=['mse'])

    # Returning the model
    return model

# Function to create the dataset
def DS():
  
  # Initializing the input
  Input = np.linspace(0, 1.5, 100).flatten()

  # Initializing the random generator
  np.random.seed(1)

  # Initializing the noise
  w = np.random.normal(-0.06, 0.06, Input.shape)

  # Defining the return variables
  Train_In = []
  Val_In = []
  Train_Out = []
  Val_Out = []

  # Generating the training and validation data
  for i in range(0,len(Input)):  
    if ((i+1) % 10) is not 0:
      Train_In.append(Input[i])
      Train_Out.append((math.sin(5*math.pi*Input[i]) + w[i]))
    else:
      Val_In.append(Input[i])
      Val_Out.append((math.sin(5*math.pi*Input[i]) + w[i]))

  # Returning the data
  return [np.asarray(Train_In).flatten(),
          np.asarray(Train_Out).flatten(),
          np.asarray(Val_In).flatten(),
          np.asarray(Val_Out).flatten()]

# Execution begins here
if __name__ == "__main__":

  # Obtaining the dataset
  [Train_X, Train_Y, Val_X, Val_Y] = DS()

  # Defining the regulariztion parameters
  Params = [0.0, 1e-6, 1e-5, 1e-4]

  # Initilizing storage variables
  models = []
  histories = []
  Train_Ps = []
  Val_Ps = []

  # Defining plotting variables
  plt.rcParams['figure.figsize'] = 20, 10
  Error_figure, Error_axis = plt.subplots(len(Params), 2)
  Error_figure.suptitle("epochs vs mse", size = 16)
  Error_figure.tight_layout(pad=3.0)
  Predict_figure, Predict_axis = plt.subplots(len(Params), 2)
  Predict_figure.suptitle("input vs output", size = 16)
  Predict_figure.tight_layout(pad=3.0)

  # Looping over the parameters
  for j in range(0, len(Params)):
    models.append([])
    histories.append([])
    Train_Ps.append([])
    Val_Ps.append([])

    # Plotting the actual noisy values
    Predict_axis[j,0].plot(Train_X,Train_Y)
    Predict_axis[j,1].plot(Val_X,Val_Y)

    for k in range(0, 3):
      
      # Initializing the model with new weights
      models[j].append(MLP(1,1,11,20,Params[j]))

      # Training the model along with capturing the history
      histories[j].append(models[j][k].fit(Train_X, Train_Y, batch_size=50,
                                           epochs=1000, verbose=0,
                                           validation_data=(Val_X, Val_Y),
                                           shuffle=True, workers=4, 
                                           use_multiprocessing=True))

      # Printing training data
      print("Regularization parameter: ", Params[j], 
            " and training attempt: ", k+1)
      print("For the final epoch, loss: ", histories[j][k].history['loss'][-1], 
            ", mse: ", histories[j][k].history['mse'][-1],
            ", val_loss: ", histories[j][k].history['val_loss'][-1], 
            ", val_mse: ",
            histories[j][k].history['val_mse'][-1])

      # Predicting training and validation data
      Train_Ps[j].append(models[j][k].predict(Train_X))
      Val_Ps[j].append(models[j][k].predict(Val_X))

      # Plotting
      Error_axis[j,0].plot(histories[j][k].history['mse'])
      Error_axis[j,1].plot(histories[j][k].history['val_mse'])
      Predict_axis[j,0].plot(Train_X,Train_Ps[j][k])
      Predict_axis[j,1].plot(Val_X,Val_Ps[j][k])

    # Printing empty line
    print("")

    # Legend and labelling the plots
    Error_axis[j,0].set_title("mse at reg. param: "+ str(Params[j]))
    Error_axis[j,0].legend(('run1', 'run2', 'run3'))
    Error_axis[j,1].set_title("val mse at reg. param: "+ str(Params[j]))
    Error_axis[j,1].legend(('run1', 'run2', 'run3'))
    Predict_axis[j,0].set_title("train pred. at reg. param: "+ str(Params[j]))
    Predict_axis[j,0].legend(('actual', 'predict1', 'predict2', 'predict3'))
    Predict_axis[j,1].set_title("val pred. at reg. param: "+ str(Params[j]))
    Predict_axis[j,1].legend(('actual', 'predict1', 'predict2', 'predict3'))