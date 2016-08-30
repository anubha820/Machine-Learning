"""
Created on Mon Mar 21 14:21:43 2016

@author: Anubha Bhargava
"""
import numpy as np
import matplotlib.pyplot as plt

## PART 1
X = np.genfromtxt('C:\Users\Anubha\Documents\Spring 2016\Homework3\X.csv',dtype="float32",delimiter=',')
Y = np.genfromtxt('C:\Users\Anubha\Documents\Spring 2016\Homework3\Y.csv',dtype="float32",delimiter=',')

# Function takes positive integer n and probability distribution w
def get_bootstrap (w,ntotal):
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

p = [0.1, 0.2, 0.3, 0.4]
n = [50,250,500]
for i in range(0,len(n)):
    result = get_bootstrap(p,n[i]) 
    plt.hist(result) #plot histogram
    plt.title('Histogram of c when n = %s'%(n[i]))    
    plt.show()

## PART 2

# Create test and train vectors
X_test, X_train = X[:183], X[183:]
Y_test, Y_train = Y[:183], Y[183:]

#Initialize variables and arrays
T = 1000
alpha = np.zeros((1000,1))
epsilon = np.zeros((1000,1))
f_all = np.zeros((500,1))
f_test = np.zeros((183,1))
f_test_boost = np.zeros((1000,183))
f_train_boost = np.zeros((1000,500))
f_train = np.zeros((500,1))
misclassified_all = np.zeros((500,1))
mistestall = np.zeros((1000,1))
mistrainall = np.zeros((1000,1))
w0=np.zeros(1000)
w1=np.zeros((9,1000))
f1=np.zeros((T,500))
f2=np.zeros((T,183))
p_2 = np.ones(500)/500.0
store_p2 = np.zeros((1000,500))
n_2 = 500
pi_1=0
pi_0=0

for t in range(0,1000): #For T=1000 iterations
    #Initialize vectors and arrays within the loop
    mu_1=np.zeros((9,1))
    mu_0=np.zeros((9,1))
    pi_1 = 0
    pi_0 = 0
    misclassified=[]
    misclassified_test = []
    misclassified_train = []
    X_bootstrap = np.zeros((500,10))
    Y_bootstrap = np.zeros((500,1))
#Get bootstrap data
    val = get_bootstrap(p_2,n_2) 
    for k in range(0,500):
        X_bootstrap[k][:] = X_train[val[k]][:]            
        Y_bootstrap[k] = Y_train[val[k]]
#Calculate pi and mu
    for i in range(0,500):
        if(Y_bootstrap[i]==1):
            pi_1 +=1
            mu_1 += ((X_bootstrap[i][1:10]).reshape(9,1))                 
        else:
            pi_0+=1
            mu_0 += ((X_bootstrap[i][1:10]).reshape(9,1))
    mu_1=mu_1/pi_1
    mu_0=mu_0/pi_0
    pi_1 = pi_1/500.0
    pi_0 = pi_0/500.0    
#Calculate Sigma    
    sigma = np.cov((X_bootstrap[:,1:10]).T)
#Calculate w using w0 and w1 given in homework statement
    w0[t] = np.log(pi_1/pi_0)-(0.5)*(np.dot((mu_1+mu_0).T,np.dot(np.linalg.inv(sigma),(mu_1-mu_0))))
    w = np.dot(np.linalg.inv(sigma),(mu_1-mu_0))
    w1[:,[t]]=w[:]
    f=np.zeros((500))
# Generate classifier
    f = np.sign(w0[t] + np.dot(X_train[:,1:],w1[:,[t]]))
# Compare with ytrain to determine the misclassified cases
# Determine epsilon and alpha
    for x in range(0,500):
        if Y_train[x]!=f[x]:
            misclassified.append(x)    
    for i in misclassified:
        epsilon[t] = epsilon[t] + p_2[i]
    alpha[t] = np.log((1-epsilon[t])/epsilon[t])*0.5
# Update weights
    sump2=0.0
    for i in range(0,500):
#Generate new probability value
        p_2[i]*=np.exp((-alpha[t]*Y_train[i]*f[i]))
        store_p2[t,i] = p_2[i] 
        sump2 = sump2+p_2[i]
    p_2=p_2/sump2 #Normalize the probability
    
#Use the classifier on the training data
    for x in range(0,500):
        f1[t][x] = np.sign(w0[t]+np.dot(X_train[x,1:],w1[:,[t]]))
#Use the classifier on the testing data
    for x in range(0,183):
        f2[t][x] = np.sign(w0[t]+np.dot(X_test[x,1:],w1[:,[t]]))

#Determine fboost by multiplying the classifier with alpha
f_test_boost[0,:] = alpha[0]*f2[0,:]
f_train_boost[0,:] = alpha[0]*f1[0,:]
for i in range(1,1000):
    f_test_boost[i,:] = alpha[i]*f2[i,:]+f_test_boost[i-1,:]
    f_train_boost[i,:] = alpha[i]*f1[i,:]+f_train_boost[i-1,:]
f_test_boost = np.sign(f_test_boost)
f_train_boost = np.sign(f_train_boost)

#Determine the misclassified training and testing cases
misclassified_train = np.zeros(1000)
misclassified_test = np.zeros(1000)
for t in range(0,1000):
    for k in range(0,500):
        if Y_train[k]!= f_train_boost[t,k]:
            misclassified_train[t]+=(1/500.0)
    for k in range(0,183):
        if Y_test[k]!=f_test_boost[t,k]:
            misclassified_test[t]+=(1/183.0)

#Use the misclassified training and testing cases to determine error
plt.plot(range(0,T),misclassified_train,label='Training')
plt.plot(range(0,T),misclassified_test,label='Test')
plt.legend(loc='upper right')
plt.title('Testing and Training Error')
plt.show()

#Plot Alpha
plt.plot(range(0,T),alpha)
plt.title('Alpha as a function of t')    
plt.show()
#Plot Epsilon
plt.plot(range(0,T),epsilon)
plt.title('Epsilon as a function of t')    
plt.show()

# Determine the testing accuracy without boosting

#Determine mu and pi
mu_1=np.zeros((9,1))
mu_0=np.zeros((9,1))
pi_1 = 0
pi_0 = 0
for i in range(0,500):
    if(Y_train[i]==1):
        pi_1 +=1
        mu_1 += ((X_train[i][1:10]).reshape(9,1))             
    else:
        pi_0+=1
        mu_0 += ((X_train[i][1:10]).reshape(9,1))
mu_1=mu_1/pi_1
mu_0=mu_0/pi_0
pi_1 = pi_1/500.0
pi_0 = pi_0/500.0
#Determine sigma
sigma = np.cov((X_train[:,1:10]).T)
w0[t] = np.log(pi_1/pi_0)-(0.5)*(np.dot((mu_1+mu_0).T,np.dot(np.linalg.inv(sigma),(mu_1-mu_0))))
w = np.dot(np.linalg.inv(sigma),(mu_1-mu_0))
w1[:,[t]]=w[:]
#Train the classifier
f=np.zeros((183))
f = np.sign(w0[t] + np.dot(X_test[:,1:],w1[:,[t]]))
#Determine the misclassified cases
misclassified_test=[]
for x in range(0,183):
    if Y_test[x]!=f[x]:
        misclassified_test.append(x)
#Generate the testing accuracy
test_accuracy = 1-(len(misclassified_test)/183.0)

#Plot pt as a function of 3 points
# Selected index values
index_1 = 50
index_2 = 100
index_3 = 120

# Plot the probabilities for the 3 different selected index values across 1000 iterations
plt.plot(range(0,T),store_p2[:,index_1],label='50')
plt.plot(range(0,T),store_p2[:,index_2],label='100')
plt.plot(range(0,T),store_p2[:,index_3],label='120')
plt.legend(loc='upper right')
plt.title('Probabilities (pt) as a function of t')    
plt.show()

## PART 3

# Initialize variables and arrays
T = 1000
alpha = np.zeros((1000,1))
epsilon = np.zeros((1000,1))
f_all = np.zeros((500,1))
f_test = np.zeros((183,1))
f_test_boost = np.zeros((1000,183))
f_train_boost = np.zeros((1000,500))
f_train = np.zeros((500,1))
misclassified_all = np.zeros((500,1))
mistestall = np.zeros((1000,1))
mistrainall = np.zeros((1000,1))
w0=np.zeros(1000)
w1=np.zeros((9,1000))
f1=np.zeros((T,500))
f2=np.zeros((T,183))
f_3 = np.zeros((500))
p_2 = np.ones(500)/500.0
store_p2 = np.zeros((1000,500))
n_2 = 500
pi_1=0
pi_0=0

for t in range(0,1000): # For T=1000 iterations
    # Initialize vectors and arrays within the loop
    mu_1=np.zeros((9,1))
    mu_0=np.zeros((9,1))
    pi_1 = 0
    pi_0 = 0
    misclassified_part3=[]
    misclassified_test = []
    misclassified_train = []
    X_bootstrap = np.zeros((500,10))
    Y_bootstrap = np.zeros((500,1))
    #Get bootstrap data
    val = get_bootstrap(p_2,n_2) 
    for k in range(0,500):
        X_bootstrap[k][:] = X_train[val[k]][:]            
        Y_bootstrap[k] = Y_train[val[k]]
#Calculate pi and mu
    for i in range(0,500):
        if(Y_bootstrap[i]==1):
            pi_1 +=1
            mu_1 += ((X_bootstrap[i][1:10]).reshape(9,1))
                 
        else:
            pi_0+=1
            mu_0 += ((X_bootstrap[i][1:10]).reshape(9,1))
        
    mu_1=mu_1/pi_1
    mu_0=mu_0/pi_0
    pi_1 = pi_1/500.0
    pi_0 = pi_0/500.0
    
#Calculate sigma, and new w for the logistic regression classifier
    w_3 = np.zeros((1,10))
    eta = 0.1
    n = 500
    for x in range(1,n): 
        sigma_1 = 1/(1+np.exp(-Y_bootstrap[x]*np.dot(X_bootstrap[x],np.transpose(w_3))))
        w_3=w_3+eta*(1-sigma_1)*(Y_bootstrap[x]*X_bootstrap[x])
    for x in range(1,n):
        f_3[x] = np.sign(np.dot(X_train[x],np.transpose(w_3)))
# Determine the misclassified classes, epsilon and alpha
    for x in range(0,500):
        if Y_train[x]!=f_3[x]:
            misclassified_part3.append(x)    
    for i in misclassified_part3:
        epsilon[t] = epsilon[t] + p_2[i]
    alpha[t] = np.log((1-epsilon[t])/epsilon[t])*0.5
# Determine new probabilities
    sump2=0.0
    for i in range(0,500):
        p_2[i]*=np.exp((-alpha[t]*Y_train[i]*f_3[i]))
        store_p2[t,i] = p_2[i] 
        sump2 = sump2+p_2[i]
    p_2=p_2/sump2
    
#Use the classifier on the new training data
    for x in range(0,500):
        f1[t][x] = np.sign(np.dot(X_train[x],np.transpose(w_3)))        
#Use the classifier on the new testing data
    for x in range(0,183):    # finding classifier
        f2[t][x] = np.sign(np.dot(X_test[x],np.transpose(w_3)))        

#Determine fboost by multiplying the classifier with alpha
f_test_boost[0,:] = alpha[0]*f2[0,:]
f_train_boost[0,:] = alpha[0]*f1[0,:]
for i in range(1,1000):
    f_test_boost[i,:] = alpha[i]*f2[i,:]+f_test_boost[i-1,:]
    f_train_boost[i,:] = alpha[i]*f1[i,:]+f_train_boost[i-1,:]
f_test_boost = np.sign(f_test_boost)
f_train_boost = np.sign(f_train_boost)

#Determine the misclassified cases for training and testing
misclassified_train = np.zeros(1000)
misclassified_test = np.zeros(1000)
for t in range(0,1000):
    for k in range(0,500):
        if Y_train[k]!= f_train_boost[t,k]:
            misclassified_train[t]+=(1/500.0)
    for k in range(0,183):
        if Y_test[k]!=f_test_boost[t,k]:
            misclassified_test[t]+=(1/183.0)

#Determine the training and testing error
plt.plot(range(0,T),misclassified_train,label='Training')
plt.plot(range(0,T),misclassified_test,label='Test')
plt.legend(loc='upper right')
plt.title('Testing and Training Error - Part 3')
plt.show()

#Determine alpha
plt.plot(range(0,T),alpha)
plt.title('Alpha as a function of t - Part 3')    
plt.show()
#Determine epsilon
plt.plot(range(0,T),epsilon)
plt.title('Epsilon as a function of t - Part 3')    
plt.show()

# Determine the testing accuracy without boosting
# Generate mu and pi
mu_1=np.zeros((9,1))
mu_0=np.zeros((9,1))
pi_1 = 0
pi_0 = 0

for i in range(0,500):
    if(Y_train[i]==1):
        pi_1 +=1
        mu_1 += ((X_train[i][1:10]).reshape(9,1))             
    else:
        pi_0+=1
        mu_0 += ((X_train[i][1:10]).reshape(9,1))
mu_1=mu_1/pi_1
mu_0=mu_0/pi_0
pi_1 = pi_1/500.0
pi_0 = pi_0/500.0

# Generate w and sigma for the logistic regression classifier
eta = 0.1
n = 500
f=np.zeros((183))
w_3 = np.zeros((1,10))
for x in range(1,n): 
    sigma_1 = 1/(1+np.exp(-Y_bootstrap[x]*np.dot(X_bootstrap[x],np.transpose(w_3))))
    w_3=w_3+eta*(1-sigma_1)*(Y_bootstrap[x]*X_bootstrap[x])
for x in range(1,n):
    f_3[x] = np.sign(np.dot(X_train[x],np.transpose(w_3)))
misclassified_test=[]
for x in range(0,183):
    if Y_test[x]!=f_3[x]:
        misclassified_test.append(x)
#Determine the testing accuracy
test_accuracy_2 = 1-(len(misclassified_test)/183.0)

# Plot pt as a function of t for 3 points
index_1 = 50
index_2 = 100
index_3 = 120

#Plot the probabiltiies of the selected indexes against the number of iterations
plt.plot(range(0,T),store_p2[:,index_1],label='50')
plt.plot(range(0,T),store_p2[:,index_2],label='100')
plt.plot(range(0,T),store_p2[:,index_3],label='120')
plt.legend(loc='upper right')
plt.title('Probabilities (pt) as a function of t - Part 3')    
plt.show()
