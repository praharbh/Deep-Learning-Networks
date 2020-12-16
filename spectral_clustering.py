"""
	Author:    Prahar Bhatt
	Created:   11.14.2020
	Center for Advanced Manufacturing, University of Southern California.
"""

# Mounting the drive
from google.colab import drive
drive.mount('/content/drive')

# Importing libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import scipy
from scipy.spatial import distance_matrix
import sklearn
from sklearn.cluster import KMeans
from google.colab import files

# Function to calculate the Graph Laplacian matrix
# and return the Eigen values and vectors
def EV(data, sigma = 0.08):
  
  # length of data
  n = len(data)

  # Calculating the Weight matrix
  W = np.exp(np.square(distance_matrix(data, data) / sigma) * -1.0)
  
  # Calculating the Degree matrix
  D = np.diag(np.sum(W, axis = 1))

  # Calculating the Graph Laplacian
  L = D - W

  # Returning the Eigen values and vectors of Graph Laplacian
  return np.linalg.eig(L)

# Execution begins here
if __name__ == "__main__":

  # Loading the file from the drive
  data = np.load('drive/My Drive/Colab Notebooks/data.npy')
  
  # Calculating the Eigen values and vectors of Graph Laplacian
  eVal, eVec = EV(data)

# Obtaining the absolute real part of Eigen values
  eVal = np.absolute(np.real(eVal))

  # Sorting the Eigen values and vectors
  sorteVal = np.sort(eVal)
  sortId = np.argsort(eVal)
  sorteVec = eVec[:,sortId]

  # Plotting the sorted Eigen values
  plt.rcParams['figure.figsize'] = 10, 5
  plt.semilogy(sorteVal[:20],'.-')
  plt.title("Semilogy plot")
  plt.xlabel("Index")
  plt.ylabel("Eigen value")

# Detemining the number of clusters
  k = 1
  for i in range(1, len(eVal)):
    if (eVal[i] - eVal[i-1]) < 1e-1:
      k += 1
    else:
      break
  print("Number of clusters: ", str(k))

  # Determing the clusters using the Eigen vector coordinates
  kMeans = KMeans(n_clusters=k).fit(sorteVec[:,:k])
  kLabel = kMeans.labels_

# Assiging data to list
  x = [data[i][0] for i in range(len(data))]
  y = [data[i][1] for i in range(len(data))]
  z = [data[i][2] for i in range(len(data))]

  # Defining plotting variables
  plt.rcParams['figure.figsize'] = 20, 10
  figure2d, axis2d = plt.subplots(1, 3)
  figure2d.suptitle("2D clustered data", size = 16)
  figure2d.tight_layout(pad=5.0)
  
  # 2D scatter plots
  axis2d[0].scatter(x, y, c = kLabel)
  axis2d[0].set_xlabel("x")
  axis2d[0].set_ylabel("y")
  axis2d[1].scatter(x, z, c = kLabel)
  axis2d[1].set_xlabel("x")
  axis2d[1].set_ylabel("z")
  axis2d[2].scatter(y, z, c = kLabel)
  axis2d[2].set_xlabel("y")
  axis2d[2].set_ylabel("z")

# 3D scatter plot
plt.rcParams['figure.figsize'] = 10, 10
figure3d = plt.figure()
figure3d.suptitle("3D clustered data", size = 16)
axis3d = figure3d.add_subplot(111, projection='3d')
axis3d.scatter(x,y,z, c = kLabel)
axis3d.set_xlabel("x")
axis3d.set_ylabel("y")
axis3d.set_zlabel("z")