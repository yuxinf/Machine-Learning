
# coding: utf-8

# In[3]:


##KNN_Classifier
get_ipython().magic('matplotlib inline')
import numpy
import matplotlib.pyplot as plt
import scipy.io
data = scipy.io.loadmat('hw1data.mat')
import random
from numpy import *

# Size of trainning data and test data
knntrainnum = 6000
knntestnum = 4000

total_index2 = []
for i in range(10000):
    total_index2.append(i)
random.shuffle(total_index2)

train_x = take(data['X'], total_index2[:knntrainnum], axis = 0)
train_x = train_x.astype(numpy.int32)
train_y = take(data['Y'], total_index2[:knntrainnum], axis = 0)
train_y = train_y.astype(numpy.int32)
test_x = take(data['X'], total_index2[knntrainnum:], axis = 0)
test_x = test_x.astype(numpy.int32)
test_y = take(data['Y'], total_index2[knntrainnum:], axis = 0)
test_y = test_y.astype(numpy.int32)

# This function does the KNN classification using Euclidean distance 
def KNNClassifier (newInput, dataset, labels, k):
    numSamples = len(dataset)
    diff = tile(newInput, (numSamples, 1)) - dataset
    squaredDist = sum(abs(diff)**2, axis=1)**0.5
    sortedDistIndices = squaredDist.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel[0]] = classCount.get(voteLabel[0], 0) + 1        
        maxCount = 0
        for key, value in classCount.items():
            if value > maxCount:
                maxCount = value
                maxIndex = key
        return maxIndex
    
# This function tests the accuarcy of the KNN classification model 
def KNNtest():
    matchCount = 0
    numTestSamples = len(test_x)
    for i in range(numTestSamples):
        predict = KNNClassifier(test_x[i], train_x, train_y, 3)
        if predict == test_y[i]:
            matchCount += 1
    accuracy = matchCount / numTestSamples
    return accuracy
accuracy = KNNtest()
print(accuracy)

