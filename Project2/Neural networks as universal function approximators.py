
# coding: utf-8

# In[16]:


get_ipython().magic('matplotlib inline')
import numpy
import matplotlib.pyplot as plt
import scipy.io
data = scipy.io.loadmat('hw2data_2.mat')
import random
import math


# In[19]:


#Set step size eta
eta = 10
#Set number of neurons
k = 8

W1 = numpy.random.random((1,k))
W2 = numpy.random.random((k,1))
b1 = numpy.random.random((k,1))
b2 = numpy.random.random((1,1))

#calculate the sigmoid equation
def sigmoid(x):
    return 1/(1+ numpy.e**(-x))

#Calculate the error
def error(data, k, W1, W2, b1, b2):
    n = len(data['X'])
    layer2 = 0
    for i in range(n):
        layer2_input = sigmoid(numpy.multiply(W1.T, data['X'][i]) + b1)
        layer2 += (sigmoid(numpy.dot(W2.T, layer2_input) + b2) - data['Y'][i][0])**2
    error = (1/(2*n))*layer2
    return(error)

#Calculate the derivation of W1 b1 W2 b2
def dev_b1w1b2w2(data, W1, W2, b1, b2, k):
    n = len(data['X'])
    
    devw1 = numpy.random.random((k,1))
    devb1 = numpy.random.random((k,1))
    devw2 = numpy.random.random((k,1))
    devb2 = numpy.random.random((1,1))
    
    devba = 0
    devwa = 0
    devbb = 0
    devwb = 0
    
    for i in range(n):
        x = data['X'][i]
        y = data['Y'][i]
        
        A1 = sigmoid(numpy.multiply(W1.T, x) + b1)
        A2 = sigmoid(numpy.dot(W2.T, A1) + b2)
   
        devb2[0][0] = (A2[0][0]-y[0])*(A2[0][0])*(1-A2[0][0])
        devw2 = devb2[0][0] * A1
        
        devb1 = (A2-y)*(A2)*(numpy.ones((1,1))-A2)*W2*(A1)*(numpy.ones((k,1))-A1)
        devw1 = devb1*x 
        
        devba += devb1
        devwa += devw1
        devbb += devb2
        devwb += devw2
        
    devba = (1/n)*devba
    devwa = (1/n)*devwa
    devbb = (1/n)*devbb
    devwb = (1/n)*devwb
    return(devba,devwa, devbb, devwb)

# update the wight and bias
def update(data, W1, W2, b1, b2, k, eta):
    n = len(data['X'])
    value = []
    e = error(data, k, W1, W2, b1, b2)
    
    #Setting accuracy threshold
    accuracy = 0.0001
    
    while e > accuracy:
        devb1, devw1, devb2, devw2 = dev_b1w1b2w2(data, W1, W2, b1, b2, k)
        W2 = W2 - eta*devw2
        b2 = b2 - eta*devb2
        W1 = W1 - eta*devw1.T
        b1 = b1 - eta*devb1
        e = error(data, k, W1, W2, b1, b2)
        #print(e)
        
    for i in range(n):
        x = data['X'][i]
        y = data['Y'][i]
        A1 = sigmoid(numpy.multiply(W1.T, x) + b1)
        A2 = sigmoid(numpy.dot(W2.T, A1) + b2)
        
        value.append(A2[0][0])
            
    return(value)


# In[18]:


#Test and graph
wellplay = update(data, W1, W2, b1, b2, k, eta)

fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)
a = axes.plot(data['X'], data['Y'],'bo', label="Original")
b = axes.plot(data['X'], wellplay,'ro', label="Predict")
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.legend(loc=4)
plt.show()

