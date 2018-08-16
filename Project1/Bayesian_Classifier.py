
# coding: utf-8

# In[4]:


##Bayesian
get_ipython().magic('matplotlib inline')
import numpy
import matplotlib.pyplot as plt
import scipy.io
data = scipy.io.loadmat('hw1data.mat')
import random

# This function processes the imported data, splits the trainning data based on their label
# and outputs a random list of trianning and testing index used for randomly spllit data 
def getdata(trainnum):
    total_index = []
    for i in range(10000):
        total_index.append(i)
    traindata_index = numpy.random.choice(total_index, trainnum, replace=False)
    testdata_index = list(set(total_index) - set(traindata_index))

    Label = []
    for i in range(10):
        Label.append([])

    for i in traindata_index:
        for j in range(10):
            if data['Y'][i][0] == j:
                Label[j].append(data['X'][i])
    return total_index, traindata_index, testdata_index, Label

# This funcion outputs mean vector and covariance matrix of multivariate Gaussian distribution,
# and its determinent and inverse, which are used later for Bayesian classifier
def getparameter(trainnum):
    total_index, traindata_index, testdata_index, Label = getdata(trainnum)
    mius = []    
    cos = []    
    detcos = []
    ps = []
    pinvs = []
    log = []

    for i in range(10):
        mi = numpy.zeros(784)
        n = len(Label[i])
        for j in range(n):
            mi=mi+Label[i][j]
        miu=mi/n
        mius.append(miu)

    for i in range(10):
        co=numpy.zeros((784,784))
        n = len(Label[i])
        for j in range(n):
            A=Label[i][j]-mius[i]
            adder = A.dot(A.T)        
            co=co+adder
        co=co/n
        co=co+1*numpy.identity(784)
        cos.append(co)
        detco = numpy.linalg.slogdet(co)
        detcos.append(detco)

    for j in range (10):
        pinvs.append(numpy.linalg.pinv(cos[j]))
        log.append(0.5*numpy.log(detcos[j][1]))
        
    return mius, cos, detcos, ps, pinvs, log


# This function uses the parameters before to do the Bayesian classification and outputs the accuracy of the test
def BayClassifier (trainnum):
    total_index, traindata_index, testdata_index, Label = getdata(trainnum)
    mius, cos, detcos, ps, pinvs, log = getparameter(trainnum)
    for i in range(len(testdata_index)):
        ps.append([])   

    for i in range (len(testdata_index)):
        for j in range(10):
            p = -0.5*numpy.dot(numpy.dot((data['X'][testdata_index[i]]-mius[j]).T, pinvs[j]),(data['X'][testdata_index[i]]-mius[j])) - log[j] 
            ps[i].append(p)

    correct = 0
    for i in range (len(testdata_index)):
        num = ps[i].index(max(ps[i]))
        if num == data['Y'][testdata_index[i]][0]:
            correct += 1
    accuracy = correct/len(testdata_index)
    return(accuracy)

# Size of tranning data
trainningdata = 6000
accuracy = BayClassifier (trainningdata)
print(accuracy)

