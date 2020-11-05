"""
  Author:    Prahar Bhatt
  Created:   10.21.2020
  Center for Advanced Manufacturing, University of Southern California.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
# %matplotlib inline 
import matplotlib.pyplot as plt
import math

# Function to generate exact solution
def Exact(a, X, k = 2):
  u = [(math.exp((a * x)/k) - 1)/(math.exp(a/k) - 1) for x in X]
  return u

# Function to create model
def MLP(Input_Dim=1,Output_Dim=1,Width=10,Depth=16):

  # Defining parameters
  Reg_Func      = keras.regularizers.l2
  Reg_Param     = 1e-5

  assert Depth > 1, 'Depth of generator must be greater than 1'

  # Defining model
  model = tf.keras.Sequential()

  # Adding first layer
  model.add(keras.layers.Dense(Width, input_shape=(Input_Dim,), 
    activation='tanh', use_bias=True,
    kernel_initializer='RandomNormal', bias_initializer='RandomNormal',
    kernel_regularizer=Reg_Func(Reg_Param)))

  # Adding remaining hidden layers
  if(Depth > 2):
      for l in range(Depth - 2):
          #model.add(keras.layers.BatchNormalization())
          model.add(keras.layers.Dense(Width, activation='tanh', 
            use_bias=True, kernel_initializer='RandomNormal', 
            bias_initializer='RandomNormal',
            kernel_regularizer=Reg_Func(Reg_Param)))
          
  # Adding output layer
  model.add(keras.layers.Dense(Output_Dim, activation=None, use_bias=True,
    kernel_initializer='RandomNormal', bias_initializer='RandomNormal',
    kernel_regularizer=Reg_Func(Reg_Param)))

  # returning model
  return model

# Function to calculate loss
def Loss(u, a, u_dot, u_ddot, k = 2, lambda_b = 15):

  # Casting variables
  k = tf.cast(tf.constant(k), tf.float64)
  lambda_b = tf.cast(tf.constant(lambda_b), tf.float64)
  Lv = tf.cast(tf.Variable(0), tf.float64)
  
  # Calculating interior loss
  for i in range(len(u)):
    Lv = tf.add(Lv, tf.math.square(tf.subtract(tf.math.multiply(a,u_dot[i]), 
      tf.math.multiply(k, u_ddot[i]))))
  Lv = tf.divide(Lv, tf.cast(tf.constant(len(u)), tf.float64))

  # Calculating boundary loss
  Lb = tf.math.multiply(lambda_b,tf.add( tf.math.square(u[0]), 
    tf.math.square(tf.subtract(u[-1], tf.cast(tf.constant(1), tf.float64)))))

  # Returning all losses
  return [tf.add(Lv, Lb), Lv.numpy(), Lb.numpy()]

# Function to train network
def Train(model, x, a, max_epoch = 1000):

  # Defining and casting training variables
  optimizer  = keras.optimizers.Adam(learning_rate=1e-2)
  a = tf.cast(tf.constant(a), tf.float64)
  loss_v = []
  loss_b = []

  # Looping epochs
  for epoch in range(max_epoch):
    
    # Tape for loss
    with tf.GradientTape() as loss_tape:

      # Tape for second derivative
      with tf.GradientTape() as ddot_tape:
        ddot_tape.watch(x)

        # Tapr for first derivative
        with tf.GradientTape() as dot_tape:
          dot_tape.watch(x)

          # Training
          u  = model(x, training=True)

        # Differentiating u
        u_dot = dot_tape.gradient(u, x)
      u_ddot = ddot_tape.gradient(u_dot, x)

      # Calculating loss
      [loss_val, v, b] = Loss(tf.cast(u, tf.float64), a, u_dot, u_ddot)

    # Differentiating loss
    grads = loss_tape.gradient(loss_val, model.trainable_variables)
    
    # Storing interior and boundary loss
    loss_v.append(v)
    loss_b.append(b)

    # applying loss gradients
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Printing loss at end of epochs
    if (epoch + 1) % max_epoch == 0:
        print("Epoch: (%d), loss: (%e)" %(epoch + 1,loss_val))

  # returning prediction and losses
  return [u.numpy(), loss_v, loss_b]

# Execution begins here
if __name__ == "__main__":

  # Defining N and a
  N = [20, 30]
  a = [3, 6, 9]

  # Defining model
  model = MLP()

  # Defining plotting variables
  plt.rcParams['figure.figsize'] = 20, 10
  u_figure, u_axis = plt.subplots(2, 3)
  u_figure.suptitle("u predicted and u exact", size = 16)
  u_figure.tight_layout(pad=5.0)
  plt.rcParams['figure.figsize'] = 20, 10
  v_figure, v_axis = plt.subplots(2, 3)
  v_figure.suptitle("interior residual", size = 16)
  v_figure.tight_layout(pad=5.0)
  plt.rcParams['figure.figsize'] = 20, 10
  b_figure, b_axis = plt.subplots(2, 3)
  b_figure.suptitle("boundary residual", size = 16)
  b_figure.tight_layout(pad=5.0)

  # Defining the variables to store output
  u_predict = {}
  u_exact = {}
  v_loss = {}
  b_loss = {}

  # Looping over all N
  for i in range(len(N)):

    # Creating input data
    x_train = tf.linspace(0, 1, N[i])

    # Looping over all a
    for j in range(len(a)):

      # Printing current training data
      print("N: " + str(N[i]) + ", a: "+ str(a[j]))

      # Obtaining predicted and exact solution
      [u_predict[(N[i],a[j])], v_loss[(N[i],a[j])],
       b_loss[(N[i],a[j])]] = Train(model, x_train, a[j])
      u_exact[(N[i],a[j])] = Exact(a[j], x_train)

      # Plotting
      u_axis[i,j].plot(x_train, u_exact[(N[i],a[j])])
      u_axis[i,j].plot(x_train, u_predict[(N[i],a[j])],'--')
      u_axis[i,j].set_title("N: " + str(N[i]) + ", a: "+ str(a[j]))
      u_axis[i,j].legend(("exact", "predicted"))
      u_axis[i,j].set_xlabel("x")
      u_axis[i,j].set_ylabel("u(x)")
      v_axis[i,j].plot(v_loss[(N[i],a[j])])
      v_axis[i,j].set_title("N: " + str(N[i]) + ", a: "+ str(a[j]))
      v_axis[i,j].set_xlabel("epoch")
      v_axis[i,j].set_ylabel("Lv")
      b_axis[i,j].plot(b_loss[(N[i],a[j])])
      b_axis[i,j].set_title("N: " + str(N[i]) + ", a: "+ str(a[j]))
      b_axis[i,j].set_xlabel("epoch")
      b_axis[i,j].set_ylabel("Lb")

      # Printing empty line
      print("")