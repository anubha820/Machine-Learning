# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 11:55:21 2016

@author: Anubha Bhargava
"""
import numpy as np
from scipy.stats.mstats import mode
import matplotlib.pyplot as plt
import matplotlib.cm as cm

xtrain = np.genfromtxt('C:\Users\Anubha\Documents\Spring 2016\hw2data\Xtrain.txt',dtype="float32",delimiter=',').T.reshape(20,5000)
xtest = np.genfromtxt('C:\Users\Anubha\Documents\Spring 2016\hw2data\Xtest.txt',dtype="float32",delimiter=',').T.reshape(20,500)
Q = np.genfromtxt('C:\Users\Anubha\Documents\Spring 2016\hw2data\Q.txt',dtype="float32",delimiter=',')
label_train = np.genfromtxt('C:\Users\Anubha\Documents\Spring 2016\hw2data\label_train.txt',dtype="int32",delimiter=',').reshape(5000,1)
yt = np.genfromtxt('C:\Users\Anubha\Documents\Spring 2016\hw2data\label_test.txt',dtype="int32",delimiter=',').reshape(500,1)

# Problem 3a
# Implement the k-NN classifier for k=1,2,3,4,5
yp = np.zeros((500,5))
for num in range(0,500): # Split the training and testing variables 500 times
    distance = np.sqrt(np.sum(np.power(xtrain-xtest[:,num:num+1],2),axis=0)).T.reshape(5000,1)
    # match distances to the labels
    labels_distance = np.concatenate((distance,label_train),axis=1)
    # sort the concatenated array
    sort_distance = np.sort(labels_distance.view('float64,float64'), axis=0).view(np.float)
    # store values for k=1,2,3,4,5
    for k in range(0,5):
        kval = sort_distance[0:k+1,1]
        # for the 500 values, store the value that occurs the most
        yp[num,k] = mode(kval)[0]

# Calculate confusion matrix and prediction accuracy. 
# Show 3 misclassified examples for k=1,3,5 and indicate the predicted class.
pred_accuracy = np.zeros((5,1))
Ck = np.zeros((10,10,5)) # calculate C for each k
misclassified=[]
# for values of k=1,2,3,4,5
for k in range(0,5): 
    for num in range(0,500):
        # Create confusion matrix using the true class and predicted class values
        Ck[yt[num,0],yp[num,k],k] += 1
        # If they aren't equal, set them as misclassified
        if yt[num,0]!=yp[num,k]:
            misclassified.append(num) 
    # Calculate the prediction accuracy
    pred_accuracy[k,0] = np.trace(Ck[:,:,k:k+1]/500)
    if k%2==0:
        # Plot the 3 misclassified images
        for i in range(0,3):
            pic = -np.dot(Q,xtest[:,misclassified[i]]).reshape(28,28)
            fig = plt.figure()
            fig.suptitle('Misclassified Example (k=%s), True Class: %s Predicted Class: %s'%(k+1,yt[misclassified[i],0],int(yp[misclassified[i],k])),fontsize=12)
            plt.imshow(pic,cm.Greys_r)
            plt.show()
    misclassified=[]

# Problem 3b
# Implement the Bayes classifier using a class-specific multivariate Gaussian distribution.
# Derive the maximum likelihood estimate for the mean and covariance for a particular class j.
# Show the answer you obtain for mean and covariance as well as the estimate for the class prior.
pi = np.zeros((10,1))
mu = np.zeros((20,10))
sigma = np.zeros((20,20,10))
n_pi = 5000 
n = 500
for num in range(0,5000):
    # Prior class
    pi[label_train[num,0],0] += 1
    # Mean
    mu[:,label_train[num,0]] += xtrain[:,num]
for num in range(0,5000):
    # Covariance
    sigma[:,:,label_train[num,0]] += np.dot((xtrain[:,num]-mu[:,label_train[num,0]]).reshape(20,1),(xtrain[:,num]-mu[:,label_train[num,0]]).reshape(20,1).T)
sigma = sigma/n
pi = pi/n_pi 
mu = mu/n    

# Implement the Bayes classifier
classifier = np.zeros((10,1))
findBayesPred = np.zeros((500,1))
for i in range(0,500):
    for j in range(0,10):
        classifier[j,0] = pi[j,0]/np.sqrt(np.linalg.det(sigma[:,:,j]))*np.exp(np.dot(np.dot((xtest[:,i]-mu[:,j]).T,np.linalg.inv(sigma[:,:,j])),(xtest[:,i]-mu[:,j]))/-2)
    # Derive the maximum likelihood estimate for the mean nd covariance 
    findBayesPred[i,0]=np.argmax(classifier)

# Create confusion matrix and prediction accuracy for Bayes Classifier
Cb = np.zeros((10,10))
misclassified_bayes = []
for k in range(0,500):
    # Create confusion matrix using the true class and predicted class values
    Cb[yt[k,0],findBayesPred[k,0]] += 1
    # If they aren't equal, set them as misclassified
    if yt[k,0]!=findBayesPred[k,0]:
        misclassified_bayes.append(k)
pred_accuracy_bayes = (np.trace(Cb)/500)

# Show three misclassified exmaples as images and show the probability 
# distribution on the 10 digits learned by the Bayes classifier for each one
for num in range(0,3):
    # Show the 3 misclassified images for Bayes Classifier
    pic = -np.dot(Q,xtest[:,misclassified_bayes[num]]).reshape(28,28)
    fig = plt.figure()            
    fig.suptitle('Misclassified Example for Bayes, True: %s Predicted: %s'%(yt[misclassified_bayes[num],0],int(findBayesPred[misclassified_bayes[num],0])), fontsize=12)
    plt.imshow(pic, cm.Greys_r)
    plt.show()

# Plot the probability distribution on the 10 digits learned by the Bayes Classifier for each one
for num in range(0,3):
    for k in range(0,10):
        classifier[k,0] = pi[k,0]/np.sqrt(np.linalg.det(sigma[:,:,k]))*np.exp(np.dot(np.dot((xtest[:,misclassified_bayes[num]]-mu[:,k]).T,np.linalg.inv(sigma[:,:,k])),(xtest[:,misclassified_bayes[num]]-mu[:,k]))/-2)
    fig = plt.figure()
    fig.suptitle('Probability Distribution for Bayes (Test Index: %s)'%(misclassified_bayes[j]), fontsize=12)
    # Normalize results
    plt.bar(np.arange(len(classifier)),classifier/np.sum(classifier))
    fig.show()

# Show the mean of each Gaussian as an image using the provided Q matrix
for num in range(0,10):
    pic = -np.dot(Q,mu[:,num]).reshape(28,28)
    fig = plt.figure()            
    fig.suptitle('Mean of Gaussian for %s'%(num), fontsize=12)
    plt.imshow(pic, cm.Greys_r)
    plt.show()

# Problem 3c
# Implement the multiclass logistic regression classifier you derived in Problem 1
stepsize = (0.1/5000)
xtrain_c = np.concatenate((xtrain,np.ones((1,5000))),axis=0)
log_likelihood = np.zeros((1000,1))
w1 = np.zeros((21,10))
w2 = np.zeros((21,10))
for k in range (0,1000): # Run the algorithm for 1000 iterations
    # Use problem 1 to compute (Part 2)
    term1 = np.sum(np.exp(np.dot(xtrain_c.T,w2)), axis=1, keepdims=True)
    for j in range(0,10): # Used 500 
        w2[:,j:(j+1)] = w1[:,j:(j+1)] + stepsize*(np.sum(xtrain_c[:,500*j:500*(j+1)], axis=1,keepdims=True)-np.sum((xtrain_c.T*np.exp(np.dot(xtrain_c.T,w1[:,j:j+1]))/term1).T, axis = 1, keepdims=True))
    w2 = w1
    # Calculate the log likelihood
    term2=0
    for i in range (0,10):
        term2 = term2 + np.sum(np.dot(xtrain_c[:,500*i:500*(i+1)].T,w1[:,i:(i+1)]))
    log_likelihood[k,0] = term2-np.sum(np.log(np.sum(np.exp(np.dot(xtrain_c.T,w1)), axis=1, keepdims=True)))

# Plot the log likelihood
fig = plt.figure()            
fig.suptitle('Log Likelihood for %s Iterations'%((k+1)), fontsize=12)
plt.plot(log_likelihood)
plt.ylabel('Log Likelihood')
plt.xlabel('# of Iterations')
fig.show()        

# Set up array
C = np.zeros((10,10))
xtest_c = np.concatenate((xtest,np.ones((1,500))),axis=0)
predlabels = np.exp(np.dot(xtest_c.T,w1))
predlabels = predlabels/np.sum(predlabels,axis=1,keepdims=True)
maxpredlabels = np.argmax(predlabels,axis=1).reshape(500,1)

misclassified_lr= []
for k in range(0,500):
    # Fill up confusion matrix 
    C[yt[k,0],maxpredlabels[k,0]] += 1
    # Find misclassified examples
    if yt[k,0]!=maxpredlabels[k,0]:
        misclassified_lr.append(k)
# Calculate the prediction accuracy
predacclr = (np.trace(C)/500)

for num in range(0,3):
# Plot the misclassified examples
    pic = -np.dot(Q,xtest[:,misclassified_lr[num]]).reshape(28,28)
    fig = plt.figure()            
    fig.suptitle('Misclassified Examples for Logistic Regression, True: %s Predicted: %s'%(yt[misclassified_lr[num],0],int(maxpredlabels[misclassified_lr[num],0])), fontsize=12)
    plt.imshow(pic, cm.Greys_r)
    plt.show()
    prob_dist = np.exp(np.dot(xtest_c[:,misclassified_lr[num]].T, w2))
    prob_dist=prob_dist/np.sum(prob_dist)
# Plot the probability distribution
for num in range(0,3):
    fig = plt.figure()
    fig.suptitle('Probability Distribution for Logistic Regression, Test Index: %s'%(misclassified_lr[num]), fontsize=12)
    plt.bar(np.arange(len(prob_dist)),prob_dist)
    fig.show()