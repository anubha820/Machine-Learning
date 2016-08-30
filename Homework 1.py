#Homework 1, Problem 3 - Anubha Bhargava

import random
import math
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
# -*- coding: utf-8 -*-

# Part 1, a - Print the vector wML
k = 392 # set the total number examples

# extract the data from the text files for x and y
x = np.genfromtxt('C:\Users\Anubha\Documents\Spring 2016\data_csv\X.txt',dtype="float32",delimiter=',')
y = np.genfromtxt('C:\Users\Anubha\Documents\Spring 2016\data_csv\y.txt',dtype="float32")

# Draw 20 random samples from 392 numbers
rand = random.sample(range(1,k-1),20)

# Delete the 20 indexes to create the training sets
xtrain = np.delete(x,rand,0)
ytrain = np.delete(y,rand,0)

# Take the 20 indexes and create the testing sets
xtest = x[rand]
ytest = y[rand]

# Calculate the vector w_ML
x_t = np.transpose(xtrain)
w_LS_1 = np.dot(np.dot(np.linalg.inv(np.dot(x_t,xtrain)),x_t),ytrain)

# Part 1, b  - Calculate the mean and standard deviation of the MAE for these 1000 tests
# Initialize arrays
mae_array=np.zeros((1000,1)) 
mean_mae=np.zeros((1000,1))
std_mae=np.zeros((1000,1))

for num in range(0,1000): # Split the training and testing variables 1000 times
    rand = random.sample(xrange(1,k-1),20) # Draw 20 samples
    ytrain = np.delete(y,rand,axis=0) #Delete these 20 from the training variables
    xtrain = np.delete(x,rand,axis=0)
    xtest = x[rand] #Set these 20 for the testing vector
    ytest = y[rand]
    x_t = np.transpose(xtrain) # Calculate the least squares solution
    w_LS_2 = np.dot(np.dot(np.linalg.inv(np.dot(x_t,xtrain)),x_t),ytrain)
    ypred = np.dot(xtest,w_LS_2) # Predict the output of the 20 testing samples
    mae = np.abs(ytest-ypred) # Calculate the mean absolute error
    mae_array[num] = np.sum(mae)/20
mean_mae = np.mean(mae_array) # Store the mean of the MAE
std_mae = np.std(mae_array) # Store the standard deviation of the MAE

#Part 2, a,b,c
# Initialize arrays
mean_error=np.zeros((4,1))
rmse=np.zeros((1000,1))
std_error=np.zeros((4,1))
error=np.zeros((20000))
mean_rmse=np.zeros((4,1))
std_rmse=np.zeros((4,1))
bins=np.zeros((4,1))
mean=np.zeros((4,1))
sigma=np.zeros((4,1))
log_likelihood = np.zeros([4,1])
n=20000 # for 20,000 errors

for p in range(0,4): # For p ranging from 1-4
    i=0
    error=np.zeros((20000))
    for num in range(0,1000): # For circulating through 1000 samples
        rand = random.sample(xrange(1,k-1),20) # Find 20 samples
        ytrain = np.delete(y,rand,axis=0) # Delete them from the training set
        xtrain = np.delete(x,rand,axis=0)
        xtest = x[rand] # Delete them from the test set
        ytest = y[rand]
        if p>0: #If p does not equal 1, we need to concatenate the matrices to include the rest of the coefficients - ex: x1^2,x2^2
            xtest = np.concatenate((xtest,np.power(xtest[:,1:7],p+1)),axis=1)
            xtrain = np.concatenate((xtrain,np.power(xtrain[:,1:7],p+1)),axis=1)
        x_t = np.transpose(xtrain) # Calculate the wML
        w_ML = np.dot(np.dot(np.linalg.inv(np.dot(x_t,xtrain)),x_t),ytrain)
        ypred = np.dot(xtest,w_ML) # Calculate the prediction using wLS
        error[i:i+20] = (ytest-ypred) #Store the error between the prediction and actual
        rmse[num] = np.sqrt(np.sum((error[i:i+20])**2)/20) # Calculate the RMSE
        i=i+20
    mean_rmse[p] = np.mean(rmse) #Store the mean of the RMSE
    std_rmse[p] = np.std(rmse) #Store the standard deviation of the RMSE
    sigma[p] = np.var(error)
    mean[p] = np.mean(error)
    plt.figure(p-1)
    plt.hist(error, bins=50, facecolor='blue') #Plot the data for a specific p
    plt.show()
    plt.title('p = %s'%(p+1))
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    log_likelihood[p] = -(n/2)*math.log(2*math.pi)-(n/2)*math.log(sigma[p])\
    -(1/(2*sigma[p]))*np.sum(np.power(error[p]-mean[p],2)) # Calculate the log likelihood using the determined variables
headers=["p","mean","std dev"]
table = [1,float(mean_rmse[0]),float(std_rmse[0])],[2,float(mean_rmse[1]),float(std_rmse[1])],\
 [3,float(mean_rmse[2]),float(std_rmse[2])], [4,float(mean_rmse[3]),float(std_rmse[3])]
print tabulate(table,headers) #Print the table of the mean and standard deviation of the RMSE
