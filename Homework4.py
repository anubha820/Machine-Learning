# -*- coding: utf-8 -*-
"""
Created on Thu Apr 07 16:06:47 2016

@author: Anubha
"""
import numpy as np
import matplotlib.pyplot as plt
import math

# This function produces a random sample
def sample (w,ntotal):
    cdf = []    
    for i in range(0,len(w)): # go through probabilities
        if i==0:
            cdf.append(w[0]) # create the CDF
        else:
            cdf.append(cdf[i-1]+w[i])
        dist = []
    for number in range(0,ntotal): # go 1 through n
        c = np.random.random() # generate a random number between 0 and 1
        if c <= cdf[0]: # if random variable is less than cdf[0]
            dist.append(0)
        for k in range(1,len(w)): 
            if cdf[k-1] < c <= cdf[k]: #if it lies between two values in cdf
                dist.append(k)
    map(int,dist)
    return dist # Return vector of c

# Given values
pi = [0.2, 0.5, 0.3]
n = 500
count_0=0
count_1=0
count_2=0
result = sample(pi,n)     
for i in range(0,n):
    if (result[i]==0):
        count_0+=1
    if (result[i]==1):
        count_1+=1
    if (result[i]==2):
        count_2+=1 

# Given mu and sigma
mu_1 = (0,0)
sigma_1 = [[1,0],[0,1]]
mu_2 = (3,0)
sigma_2 = [[1,0],[0,1]]
mu_3 = (0,3)
sigma_3 = [[1,0],[0,1]]

# Generate a random multivariate Gaussian distribution
n1 = np.random.multivariate_normal(mu_1,sigma_1,count_0)
n2 = np.random.multivariate_normal(mu_2,sigma_2,count_1)
n3 = np.random.multivariate_normal(mu_3,sigma_3,count_2)

# Concatenate the matrices
impdata = np.concatenate([n1,n2,n3])

# Part 1

# Perform Kmeans clustering
#Inputs:
# kval=2,3,4,5
# n = 500 observations
# imported dataset
# number of vectors (2 for first problem, 10 for second)
def kmeans(kval,n,data,numvec):
    mu = np.zeros((kval,numvec)) # K mean vectors
    index = np.random.randint(0,len(data),20)
    for r in range(0,kval): # generate random mu
        mu[r,:] = data[index[r],:]
    classes = np.zeros([n])
    obj_func = np.zeros((20,1))
    objval = np.zeros((20,n))
    closest_dist = np.zeros((kval))
    # 20 iterations
    for j in range(0,20): # for 20 iterations
        count = np.zeros(kval)
        new_mu = np.zeros((kval,numvec))
        for i in range(0,n): # for 500 samples
            for k in range(0,kval): # for specified K value
                #compare 0-500 mu to 0-k data values and find closest distance
                closest_dist[k]=np.square(np.linalg.norm(data[i,:]-mu[k,:])) # v = 10x1682, u = 943x10
            classes[i] = np.argmin(closest_dist)
            count[classes[i]]+=1
            # generate a new mu
            new_mu[classes[i],:] = new_mu[classes[i],:] + data[i,:]
        for k in range(0,kval):
            # divide the new mu by count array
            new_mu[k,:]/=count[k]
        for i in range(n):
            # generate the objective function
            objval[j,i]=objval[j,i]+np.square(np.linalg.norm(data[i,:]-new_mu[classes[i],:]))
        obj_func[j] = objval[j].sum()
        mu=new_mu
    A = classes.reshape(n,1)
    return obj_func.reshape(20),A,mu
    
# Plot the clusters 
K = (2,3,4,5)
obj=np.zeros([4,20])
n = 500
for d in range(0,4):
    obj[d,:], A, mu = kmeans(K[d],n,impdata,2)
    plt.scatter(impdata[:,0],impdata[:,1],c=(A))
    plt.plot(mu[:,0], mu[:,1], 'b*', markersize = 22)
    plt.title('Clusters,k=%s'%(K[d]))
    plt.show()
# Plot the objective function
for d in range(0,4):
    plt.plot(obj[d],label='k=%s'%(K[d]))
plt.legend(loc='upper right')
plt.title('Objective Function')
plt.show()

# Part 2

# Import the user movie dataset
ratings = np.genfromtxt(r'C:\Users\Anubha\Documents\Spring 2016\Homework4_ML\ratings.txt',dtype="float32",delimiter=",")
ratings_test = np.genfromtxt(r'C:\Users\Anubha\Documents\Spring 2016\Homework4_ML\ratings_test.txt',dtype="float32",delimiter=',')
movies = open(r'C:\Users\Anubha\Documents\Spring 2016\Homework4_ML\movies.txt','r')
movies_lines = movies.readlines()
ratings[:,0:2] = ratings[:,0:2]-1
ratings_test[:,0:2] = ratings_test[:,0:2]-1

# Given values
var = 0.25
d = 10.0
lambdaval = 10.0
identity=np.identity(10)
# from 943 users on 1682 movies
M = np.zeros((943,1682))
Mold = np.zeros((943,1682))

# create M matrix from ratings file (some are missing - those elements are zero)
for i in range(0,95000):
    M[(int(ratings[i,0])),(int(ratings[i,1]))]=int(ratings[i,2])
Mold = M

# Train the model 100 iterations
# Map the relevant dot product to the closest integer from 1 to 5

v = np.zeros([10,1682])
for k in range(0,1682):
    v[:,k] = np.random.multivariate_normal(np.zeros(10.0),(1/10.0)*np.identity(10))

term1 = lambdaval*var*identity
u = np.zeros([943,10])
length = 10
rmse = np.zeros([length]) 
log_likelihood = np.zeros([length])
# Go through the 100 iterations
for d in range(0,length):
    for i in range(0,943):
    # find corresponding user
        movie_user_id = ratings[np.where(ratings[:,0]==i),1] 
        # find corresponding movies for the user
        term2_u = np.zeros([10,10])
        term3_u = np.zeros([10])
        for j in range(0,movie_user_id.size):
            # Calculate u
            v_inst = v[:,int(movie_user_id[0,j])]
            term2_u = term2_u + np.dot(v_inst.reshape(10,1),v_inst.reshape(10,1).T)
            term3_u = term3_u + np.dot(Mold[i,int(movie_user_id[0,j])],v_inst)
        u[i,:] = np.dot(np.linalg.inv(term1 + term2_u),term3_u)
    for k in range(0,1682):
        # for a specific movie, find the users which rated it
        user_id = ratings[np.where(ratings[:,1]==k),0]
        term2_v = np.zeros([10,10])
        term3_v = np.zeros([10])
        # calculate v
        for z in range(0,user_id.size): 
            u_inst = u[int(user_id[0,z])] 
            term2_v = term2_v + np.dot(u_inst.reshape(10,1),u_inst.reshape(10,1).T)
            term3_v = term3_v + np.dot(Mold[int(user_id[0,z]),k],u_inst)
        v[:,k] = np.dot(np.linalg.inv(term1 + term2_v),term3_v)
    # Calculate the dot product for the two vectors
    M = np.rint(np.dot(u,v))
    # If it is greater than 5, round down to 5
    # If it is less than 1, round up to 1
    for i in range(0,943):
        for j in range(0,1682):
            if M[i,j] < 1:
                M[i,j] = 1.0
            if M[i,j] > 5:
                M[i,j] = 5.0
    # Calculate the log likelihood
    sigma = np.identity(10)/lambdaval
    term4 = -((1682*943)/2.0)*math.log(2*math.pi) - ((1682*943)/2.0)*math.log(1.0/np.linalg.det(sigma))
    term3 = 0
    for i in range(943):
        for j in range(1682):
            if(M[i,j]>0):
                term3 =  term3 - lambdaval/2.0 * (M[i,j] - np.dot(u[i,:].T,v[:,j]))
    log_likelihood_u = -(943/2.0)*np.log(2*np.pi) - (943/2.0)*np.log(1.0/np.linalg.det(sigma)) - (1.0*np.linalg.det(sigma)/(2))*np.sum(np.diag(np.dot(u,u.T)))
    log_likelihood_v = -(1682/2.0)*np.log(2*np.pi) - (1682/2.0)*np.log(1.0/np.linalg.det(sigma)) - (1.0*np.linalg.det(sigma)/(2))*np.sum(np.diag(np.dot(v,v.T)))
    log_likelihood[d] = log_likelihood_u + log_likelihood_v + term3 + term4
    # Calculate the RMSE    
        
    for i in range(len(ratings_test)):
        rmse[d] = rmse[d] + (1.0/5000)*np.square(ratings_test[i,2] - M[ratings_test[i,0]-1,ratings_test[i,1]-1])
    rmse[d]=np.sqrt(rmse[d])
    # Print iteration
    print d

# Plot the RMSE and Log Likelihood
plt.plot(rmse[1:100])
plt.xlim((1, length))
plt.title('RMSE')
plt.show()
plt.plot(log_likelihood)
plt.xlim((1, length))
plt.title('Log Likelihood')
plt.show()

# pick 3 reasonably well-known movies from the list:
# Index 0 - Toy Story (1995)
# Index 142 - Sound of Music, The (1965)
# Index 402 - Batman (1989)
movie_array = [0, 142, 402]
results=np.zeros((3,5))
distance=np.zeros((3,5))
z=0
# Compare the selected movie to the list of movies
# Get the closest5 results and store them
for j in movie_array:
    select_movie = movies_lines[j]
    compare_mov=[]
    for i in range(0,1682):
        if(not(j==i)):
            compare_mov.append(np.linalg.norm(v[:,j]-v[:,i]))
    sort = np.argsort(compare_mov)
    distance[z,:] = np.sort(compare_mov)[0:5]
    results[z,:] = sort[0:5]
    z=z+1
moviename=[]
# Find the movie names for these five results
for i in range(0,3):
    for j in range(0,5):
        moviename.append(movies_lines[int(results[i,j])])
output_movie = np.vstack((distance[0,:], moviename[0:5]))
output_movie2 = np.vstack((distance[1,:], moviename[5:10]))
output_movie3 = np.vstack((distance[2,:], moviename[10:15]))
# Output the distance, indices and names of the movies
print movies_lines[movie_array[0]], output_movie, movies_lines[movie_array[1]], output_movie2, movies_lines[movie_array[2]], output_movie3

# Perform Kmeans clustering on the u and v vectors
kval = 20
n = 943
numvec = 10
kmean_u, A2, mu2 = kmeans(kval,n,u,numvec)
n = 1682
kmean_v, A3, mu3 = kmeans(kval,n,np.transpose(v),numvec)

# Gather the number of users for the clusters that occur the most
ua,uind = np.unique(A2,return_inverse=True)
count_u = np.bincount(uind)
count_users = np.argsort(count_u)
count_num = np.sort(count_u)
users_u5 = count_users[-5:]
u_num5 = count_num[-5:]

# Gather the 5 clusters that have the most data for movies
ua,uind = np.unique(A3,return_inverse=True)
count_v = np.bincount(uind)
count_users_v = np.argsort(count_v)
count_num_v = np.sort(count_v)
users_v5 = count_users_v[-5:]
v_num5 = count_num_v[-5:]

centroids_u = np.vstack([users_u5, u_num5])
centroids_v = np.vstack([users_v5, v_num5])

# Output the result
print "5 centroids that have the most data in u:", centroids_u
print "5 centroids that have the most data in v:", centroids_v

# For the 5 centroids for u, give the 10 movies with the largest 
# dot product
for i in range(0,5):
    movies_u = np.dot(mu2[users_u5[i]],v)
    dot = np.sort(movies_u)[-10:]
    x = np.argsort(movies_u)[-10:]
    moviename=[]
    for j in range(0,10):
        moviename.append(movies_lines[x[j]])
    concat = np.vstack([dot, moviename])
    print "\n", "Centroid:", centroids_u[0,i], "The ten movies with the largest dot product:", concat

# For the 5 centroids for v, give the 10 movies with
# the smallest euclidean distance
for i in range(0,5):
    index=[]
    val=[]
    moviename=[]
    euclid=np.zeros([1682])
    for j in range(0,1682):
        euclid[j]=np.linalg.norm(v[:,j]-mu3[centroids_v[0,i]].T)
    val = np.sort(euclid)[:10]
    index = np.argsort(euclid)[:10]
    print val 
    print index
    print euclid
    for j in range(0,10):
        moviename.append(movies_lines[index[j]])
    concat = np.vstack([val, moviename])
    print "\n","Centroid:", centroids_v[0,i],"The ten movies with the smallest Euclidean distance: ", concat