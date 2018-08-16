
# coding: utf-8

# In[45]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Importing the dataset
te_dataset = pd.read_csv('reviews_te.csv')
tr_dataset = pd.read_csv('reviews_tr.csv')


# In[46]:


# Unigram representation
def tf(data):
    onedict = {}
    for j in range(len(data.split(" "))):
        if data.split(" ")[j] not in onedict.keys():
            onedict[data.split(" ")[j]] = 1
        else: 
            onedict[data.split(" ")[j]] += 1
        
        # data lifting
        onedict["WP"] = 1
    return(onedict)


# In[47]:


# idf representation for training data
def idf(data):
    alldict = {}
    for i in range(len(data["text"])):
        sp = data["text"][i].split(" ")
        n = 0
        for j in range(len(sp)):
            if sp[j] not in alldict.keys():
                alldict[sp[j]] = 1
            else: 
                n += 1
        if n > 0:
            alldict[sp[j]] = alldict[sp[j]] + 1      
                
    for key in alldict:
        alldict[key] = np.log10(len(data["text"]) / alldict[key])
        
    return(alldict)

# idf representation for testing data
def idf_test(data):
    alldict = {}
    for i in range(len(data["text"])):
        sp = data["text"][i].split(" ")
        n = 0
        for j in range(len(sp)):
            if sp[j] not in alldict.keys():
                alldict[sp[j]] = 1
            else: 
                n += 1
        if n > 0:
            alldict[sp[j]] = alldict[sp[j]] + 1      
                
    for key in alldict:
        alldict[key] = np.log10(len(data["text"]) / alldict[key])
        
    return(alldict)

# idf-tf representation
def tfidf(data, idff):
    tfidfdict = {}
    for j in range(len(data.split(" "))):
        if data.split(" ")[j] not in tfidfdict.keys():
            tfidfdict[data.split(" ")[j]] = 1
        else: 
            tfidfdict[data.split(" ")[j]] += 1
            
    tfidfdict = {k: tfidfdict[k]*idff[k] for k in tfidfdict}
    
    # data lifting
    tfidfdict["WP"] = 1
        
    return(tfidfdict)


# In[48]:


# Bigram representation
def bigram(data):
    bidict = {}
    for j in range(len(data.split(" "))-1):
        tu = (data.split(" ")[j], data.split(" ")[j+1])
        if tu not in bidict.keys():
            bidict[tu] = 1
        else: 
            bidict[tu] += 1
            
        # data lifting
        bidict["WP"] = 1
        
    return(bidict)


# In[49]:


# update weight for unigram
def calweight(data):
    W_final = {}
    W_final = summation1(W_final, data)
    W_final = summation2(W_final, data)
    for k in W_final.keys():
        W_final[k] = 1/(len(data) +1) * W_final[k]
    return(W_final)
         
def summation1(W_final, data):
    index = [i for i in range(len(data))]
    random.shuffle(index)
    for i in index:
        tff = tf(data["text"][i])
        #bigramm = bigram(data["text"][i])
        dot_product = sum(tff[k]*W_final.get(k, 0) for k in tff)
        y = (data["label"][i]-0.5)*2
        
        if dot_product*y <= 0:
            for key in tff.keys():
                if key not in W_final.keys():
                    W_final[key] = tff[key]
                else: 
                    W_final[key] += y*tff[key]
                
    return W_final

def summation2(W_final, data):
    W_sum = {}
    index = [i for i in range(len(data))]
    random.shuffle(index)
    for i in index:
        tff = tf(data["text"][i])
        #bigramm = bigram(data["text"][i])
        dot_product = sum(tff[k]*W_final.get(k, 0) for k in tff)
        y = (data["label"][i]-0.5)*2
        
        if dot_product*y <= 0:
            for key in tff.keys():
                if key not in W_final.keys():
                    W_final[key] = tff[key]
                else: 
                    W_final[key] += y*tff[key]
    
        for kk in W_final.keys():
            if kk not in W_sum.keys():
                W_sum[kk] = W_final[kk]
            else: 
                W_sum[kk] += W_final[kk]
                
    return W_sum


# test the prediction accuracy for unigram representation
def test(weight, testdata):
    correct = 0
    for i in range(len(testdata)):
        y = (testdata["label"][i]-0.5)*2
        testdata_weight = tf(testdata["text"][i])
        dot_product = sum(testdata_weight[k]*weight.get(k, 0) for k in testdata_weight)
        
        if dot_product * y >= 0:
            correct += 1
    accuracy = (correct/len(testdata))*100        
    return(accuracy)


# In[50]:


# update weight for idf
def calweight_idf(data):
    W_final = {}
    W_final = summation3(W_final, data)
    W_final = summation4(W_final, data)
    for k in W_final.keys():
        W_final[k] = 1/(len(data) +1) * W_final[k]
    return(W_final)
         
def summation3(W_final, data):
    index = [i for i in range(len(data))]
    random.shuffle(index)
    for i in index:
        tfidff = tfidf(data["text"][i], idff)
        #bigramm = bigram(data["text"][i])
        dot_product = sum(tfidff[k]*W_final.get(k, 0) for k in tfidff)
        y = (data["label"][i]-0.5)*2
        
        if dot_product*y <= 0:
            for key in tfidff.keys():
                if key not in W_final.keys():
                    W_final[key] = tfidff[key]
                else: 
                    W_final[key] += y*tfidff[key]
                
    return W_final

def summation4(W_final, data):
    W_sum = {}
    index = [i for i in range(len(data))]
    random.shuffle(index)
    for i in index:
        tfidff = tfidf(data["text"][i], idff)
        dot_product = sum(tfidff[k]*W_final.get(k, 0) for k in tfidff)
        y = (data["label"][i]-0.5)*2
        
        if dot_product*y <= 0:
            for key in tfidff.keys():
                if key not in W_final.keys():
                    W_final[key] = tfidff[key]
                else: 
                    W_final[key] += y*tfidff[key]
    
        for kk in W_final.keys():
            if kk not in W_sum.keys():
                W_sum[kk] = W_final[kk]
            else: 
                W_sum[kk] += W_final[kk]
                
    return W_sum   


# test the prediction accuracy for idf representation
def test_idf(weight, testdata):
    correct = 0
    for i in range(len(testdata)):
        y = (testdata["label"][i]-0.5)*2
        
        testdata_weight = tfidf(testdata["text"][i], idff_test)
        
        dot_product = sum(testdata_weight[k]*weight.get(k, 0) for k in testdata_weight)
        
        if dot_product * y >= 0:
            correct += 1
    accuracy = (correct/len(testdata))*100        
    return(accuracy)


# In[51]:


# update weight for bigram
def calweight_bigram(data):
    W_final = {}
    W_final = summation5(W_final, data)
    W_final = summation6(W_final, data)
    for k in W_final.keys():
        W_final[k] = 1/(len(data) +1) * W_final[k]
    return(W_final)
         
def summation5(W_final, data):
    index = [i for i in range(len(data))]
    random.shuffle(index)
    for i in index:
        bigramm = bigram(data["text"][i])
        dot_product = sum(bigramm[k]*W_final.get(k, 0) for k in bigramm)
        y = (data["label"][i]-0.5)*2
        
        if dot_product*y <= 0:
            for key in bigramm.keys():
                if key not in W_final.keys():
                    W_final[key] = bigramm[key]
                else: 
                    W_final[key] += y*bigramm[key]
                
    return W_final

def summation6(W_final, data):
    W_sum = {}
    index = [i for i in range(len(data))]
    random.shuffle(index)
    for i in index:
        bigramm = bigram(data["text"][i])
        dot_product = sum(bigramm[k]*W_final.get(k, 0) for k in bigramm)
        y = (data["label"][i]-0.5)*2
        
        if dot_product*y <= 0:
            for key in bigramm.keys():
                if key not in W_final.keys():
                    W_final[key] = bigramm[key]
                else: 
                    W_final[key] += y*bigramm[key]
    
        for kk in W_final.keys():
            if kk not in W_sum.keys():
                W_sum[kk] = W_final[kk]
            else: 
                W_sum[kk] += W_final[kk]
                
    return W_sum 

# test prediction accuracy for the bigram representation
def test_bigram(weight, testdata):
    correct = 0
    for i in range(len(testdata)):
        y = (testdata["label"][i]-0.5)*2
        testdata_weight = bigram(testdata["text"][i])
        dot_product = sum(testdata_weight[k]*weight.get(k, 0) for k in testdata_weight)
        
        if dot_product * y >= 0:
            correct += 1
    accuracy = (correct/len(testdata))*100        
    return(accuracy)


# In[53]:


num_train = 100000
num_test = len(te_dataset)

# Test the accuracy for Unigram
unigram_weight = calweight(tr_dataset[:num_train])
accuracy_unigram = test(unigram_weight, te_dataset[:num_test])
print(accuracy_unigram)

# Test the accuracy for idf
#idff = idf(tr_dataset[:num_train])
#idff_test = idf_test(te_dataset[:num_test])
#idf_wight = calweight_idf(tr_dataset[:num_train])
#accuracy_idf = test_idf(idf_wight, te_dataset[:num_train])
#print(accuracy_idf)

# Test the accuracy for Bigram
#bigram_weight = calweight_bigram(tr_dataset[:num_train])
#accuracy_biagram = test_bigram(bigram_weight, te_dataset[:num_test])
#print(accuracy_biagram)

