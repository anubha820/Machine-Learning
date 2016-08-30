# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 17:52:43 2016

@author: Anubha
"""
import numpy as np
import matplotlib.pyplot as plt

scores = np.genfromtxt('C:\Users\Anubha\Documents\Spring 2016\Homework5 ML\cfb2015scores.csv',dtype="float32",delimiter=',')
legend = np.genfromtxt('C:\Users\Anubha\Documents\Spring 2016\Homework5 ML\legend.txt',dtype="|S",delimiter=',')

scores[:,0] = scores[:,0]-1
scores[:,2] = scores[:,2]-1

#    scores[i,0] # Team 1 index
#    scores[i,1] # Team 1 points
#    scores[i,2] # Team 2 index
#    scores[i,3] # Team 2 points

M = np.zeros([759,759])
# game i
# j1 = team 1 index
# j2 = team 2 index

for i in range(0,len(scores)): # number of games
    j1 = scores[i,0] # team1
    j2 = scores[i,2] #team2
    points_j1 = scores[i,1]
    points_j2 = scores[i,3]
    if scores[i,3] < scores[i,1]: # team 1 wins
        M[j1,j1] = M[j1,j1] + 1 + points_j1/(points_j1+points_j2)
        M[j2,j2] = M[j2,j2] + points_j2/(points_j1+points_j2)
        M[j1,j2] = M[j1,j2] + points_j2/(points_j1+points_j2)
        M[j2,j1] = M[j2,j1] + 1 + points_j1/(points_j1+points_j2)
    else: # team 2 wins
        M[j1,j1] = M[j1,j1] + points_j1/(points_j1+points_j2)
        M[j2,j2] = M[j2,j2] + 1 + points_j2/(points_j1+points_j2)
        M[j1,j2] = M[j1,j2] + 1 + points_j2/(points_j1+points_j2)
        M[j2,j1] = M[j2,j1] + points_j1/(points_j1+points_j2)   
for j in range(0,759):
    M[j,:] = M[j,:]/np.sum(M[j,:])
# normalize each row so that it sums to 1
#t = [10, 100, 1000, 2500]
#wt = 1x759 state vector at step t
    
eigenvalues,eigenvectors = np.linalg.eig(M.T)
biggest_eig_index = np.argmax(eigenvalues)
#biggest_eig_index = np.argsort(eigenvalues)[::-1][0]
u = eigenvectors[:,biggest_eig_index]
w_inf = np.transpose(u)/sum(u)
#evals_large, w_inf = largest_eigh(M, eigvals=(758,758))

#w_inf = w_inf.reshape(759)

l1_dist = np.zeros(2501)
w = np.zeros((2501,759))
w[0,:] = np.ones(759)*(1/759.0)
l1_dist[0] = np.sum(abs(w[0,:]-w_inf))

for t in range(1,2501):
    w[t,:] = np.dot(w[t-1,:],M)
    val = abs(w[t,:] - w_inf)
    l1_dist[t] = np.sum(val)

plt.plot(l1_dist)
plt.show()

l1_dist[2500]

sort_10 = np.argsort(w[10,:])[::-1][0:25]
sort_100 = np.argsort(w[100,:])[::-1][0:25]
sort_1000 = np.argsort(w[1000,:])[::-1][0:25]
sort_2500 = np.argsort(w[2500,:])[::-1][0:25]

sort_10_names=[]
sort_100_names=[]
sort_1000_names=[]
sort_2500_names=[]
for i in range(0,25):
    sort_10_names.append(legend[sort_10[i]])
    sort_100_names.append(legend[sort_100[i]])
    sort_1000_names.append(legend[sort_1000[i]])
    sort_2500_names.append(legend[sort_2500[i]])

# Find the biggest eigenvalue of M
# Find the eigenvector that corresponds to the biggest eigenvalue 
# u is the eigenvector
# lambda is the eigenvalue
#np.transpose(M)*eigenvector = eigenvalue*eigenvector

# returns w - eigenvalues
# returns v, eigenvector v[:,i] corresponding to eigenvalue w[i]
# Plot ||wt-winf||1 as a function of t for t=1,...2500 
# What is the value of ||w2500-winf||1

## Part 2
# 8447 documents from the NYT
# vocab size is 3012 words
# Create matrix X where Xij is the number of times word i appears in document j
# X is 3012 by 8447

faces = np.genfromtxt(r'C:\Users\Anubha\Documents\Spring 2016\Homework5 ML\faces.csv',delimiter=',',dtype="float32")

n = 1024
m = 1000
K = 25
# first number is index, second number is count
W = np.zeros((n,K))
H = np.zeros((K,m))

for i in range(0,n):
    for j in range(0,K):
        W[i,j] = np.random.uniform()
for i in range(0,K):
    for j in range(0,m):
        H[i,j] = np.random.uniform()

eu_total = np.zeros((200))
for t in range(0,200):
    H = H*(np.dot(np.transpose(W),faces))/(np.dot(np.transpose(W),np.dot(W,H)))
    W = W*(np.dot(faces,np.transpose(H)))/(np.dot(W,np.dot(H,np.transpose(H))))
    eu_obj=np.power((faces-np.dot(W,H)),2)
    eu_total[t]=np.sum(eu_obj)

plt.plot(eu_total)
plt.title('Objective Function')
plt.show()

# Pick 10 col in W randomly
# Row i of H, select largest element
# That element corresponds to the column in X
# reshape column to 32,32
# plot

for i in range(0,10):
    w = W[:,i].reshape((32,32),order='F')
    col = np.argmax(H[i,:])
    x = faces[:,col].reshape((32,32),order='F')
    plt.figure()
    plt.title('W matrix, i=%s'%i)
    plt.imshow(w)
    plt.figure()
    plt.title('Original data, i=%s'%i)
    plt.imshow(x)

## Part 3
X = np.zeros((3012,8447))
nytvocab = np.genfromtxt(r'C:\Users\Anubha\Documents\Spring 2016\Homework5 ML\nytvocab.dat',delimiter=',',dtype="|S")
with open (r'C:\Users\Anubha\Documents\Spring 2016\Homework5 ML\nyt_data.txt','r') as file:
    x = file.readlines()
    y = np.array([elem.split(',') for elem in x])
    
for i in range(0,len(y)):
    for j in range(0,len(y[i])-1):
        X[int(y[i][j].split(':')[0])-1,i] = int(y[i][j].split(':')[1])
        
n = 3012
m = 8447
K = 25
# first number is index, second number is count
W = np.zeros((n,K))
H = np.zeros((K,m))

for i in range(0,n):
    for j in range(0,K):
        W[i,j] = np.random.uniform()+1e-16
for i in range(0,K):
    for j in range(0,m):
        H[i,j] = np.random.uniform()+1e-16

eu_total = np.zeros((200))
W_norm = np.zeros((n,K))
H_norm = np.zeros((K,m))
for t in range(0,200):
    for i in range(0,25):
        W_norm[:,i] = W[:,i]/(sum(W[:,i])+1e-16)
    new_X = X/(np.dot(W,H)+1e-16)
    H = H*(np.dot(np.transpose(W_norm),new_X))
    for i in range(0,25):
        H_norm[i,:] = H[i,:]/(sum(H[i,:])+1e-16)
    new_X = X/(np.dot(W,H)+1e-16)
    W = W*(np.dot(new_X,np.transpose(H_norm)))
    new_X = X/(np.dot(W,H)+1e-16)
    eu_obj=X*np.log(1/(np.dot(W,H)+1e-16))+np.dot(W,H)
    eu_total[t]=np.sum(eu_obj)

plt.figure()
plt.plot(eu_total)
plt.title('Objective Function, Part 2')
plt.show()

for j in range(0,K):
    W[:,j] = W[:,j]/np.sum(W[:,j])

ind = np.zeros((10))
value=np.zeros((10))
for i in range(0,10):
    words=[]
    ind = np.argsort(W[:,i])[-10:]
    value = np.sort(W[:,i])[-10:]
    for j in range(0,10):
        words.append(nytvocab[ind[j]])
    concat = np.vstack([value, words])
    print concat