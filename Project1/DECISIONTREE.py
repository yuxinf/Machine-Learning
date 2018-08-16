import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.stats import mode
from random import sample

class DecisionTree(object):
    F=None
    T=None
    label=None
    L=None
    R=None


def uncertain(labels):
    # Gini index
    size=len(labels)
    gini_index=1
    for i in range(10):
        num=np.count_nonzero(labels==i)
        P=num/size
        gini_index=gini_index-P**2
    return gini_index
    

def divide_data(features,labels,size_to_test):
    test_indices = sample(range(len(labels)),size_to_test)
    train_indices = np.delete(range(len(labels)), test_indices)       
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    test_features = features[test_indices]
    test_labels = labels[test_indices]
    return train_features,train_labels,test_features,test_labels

def find_FT(features,labels,greed=3):
    F=None
    T=128 # the grayscale is from 0 to 255, so pick the middle as a begin point
    U=1 # U <= 1
    for i in range(greed):
        for j in range(784):
            the_feature=np.take(features,j,axis=1) # extract the degree of all features
            if np.max(the_feature) >= T:
                if np.min(the_feature ) < T:
                    left_labels = labels[np.where(the_feature < T)[0]] # divide
                    right_labels = labels[np.where(the_feature >= T)[0]] # divide
                    U_L=left_labels.shape[0]/labels.shape[0]*uncertain(left_labels) # gini index of left part
                    U_R=right_labels.shape[0]/labels.shape[0]*uncertain(right_labels) # gini index of right part
                    U_LR=U_L+U_R
                    if U_LR < U: 
                        U=U_LR # update
                        F=j # update
        the_feature=np.take(features,F,axis=1)
        for j in range(np.min(the_feature)+1,np.max(the_feature)):
            left_labels = labels[np.where(the_feature < T)[0]]
            right_labels = labels[np.where(the_feature >= T)[0]]
            U_L=left_labels.shape[0]/labels.shape[0]*uncertain(left_labels)
            U_R=right_labels.shape[0]/labels.shape[0]*uncertain(right_labels)
            U_LR=U_L+U_R
            if U_LR < U:
                U=U_LR
                T=j 
        return F,T

def build_tree(features,labels,depth=0,K=20):
    # build the decision tree with training features and labels.
    uncertainty = uncertain(labels)
    if uncertainty == 0  or depth==K:
        node = DecisionTree()
        node.label=mode(labels)[0][0]
        return node # stop to divide the branch when it reached a certain depth or all the elements are same.

    F,T=find_FT(features,labels) # find the threshhold and which degree to compare
    the_feature=np.take(features,F,axis=1)
    left_labels = labels[np.where(the_feature < T)[0]] # divide
    left_features = features[np.where(the_feature < T)[0]] # divide
    right_labels = labels[np.where(the_feature >= T)[0]] # divide
    right_features = features[np.where(the_feature >= T)[0]] # divide
    node = DecisionTree() 
    node.depth=depth 
    node.F=F
    node.T=T
    node.L = build_tree(left_features, left_labels,depth=depth+1,K=K) #  continue to stretch the tree
    node.R = build_tree(right_features, right_labels,depth=depth+1,K=K) # continue to stretch the tree
    
    return node

def classify(classifier,feature):
    # given a classifier and a x, output the predicted y
    if classifier.L == None:
        if classifier.R == None:
            return classifier.label
    F, T = classifier.F, classifier.T
    if feature[F] < T:
        return classify(classifier.L,feature)
    else:
        return classify(classifier.R,feature)
    
def evaluate(features,labels,classifier):
    # building classifiers according to a list of K, and plot the result of error.
    true_number=0
    for i in range(labels.shape[0]):
        result = classify(classifier, features[i])
        if result == labels[i]:
            true_number=true_number + 1
    acc=true_number/len(labels)   
    error=1-acc
    return error

def Build_classifier_based_on_K_and_Plot(train_features,train_labels,test_features,test_labels):
    K_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,50,100]
    error_train=[]
    error_test=[]
    for i in K_list:
        classifier=build_tree(train_features,train_labels,K=i) # build classifier
        error_test.append(evaluate(test_features,test_labels,classifier)) # save the result of testing data
        error_train.append(evaluate(train_features,train_labels,classifier)) # save the result of training data
    plt.figure() #plotting
    plt.plot(K_list,error_train,'r',label='train data')
    plt.plot(K_list,error_test,'g',label='test data')
    plt.xlabel('K')
    plt.ylabel('error rate %')
    plt.title('error vs K')
    plt.legend(loc=1)
    plt.show()
    
if __name__ == "__main__":
    data = scipy.io.loadmat('hw1data') # load data
    features=data['X'] # load X
    labels=data['Y'] # load Y
    train_features,train_labels,test_features,test_labels=divide_data(features,labels,size_to_test=1000) # randomly split data into 1000 pieces for testing and 9000 pieces for training.
    Build_classifier_based_on_K_and_Plot(train_features,train_labels,test_features,test_labels) # building classifiers according to a list of K, and plot the result of error.