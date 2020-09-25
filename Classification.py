#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[2]:


def loadData():
    data = pd.read_csv("E202-COMP7117-TD01-00 - classification.csv")
    
    if (data.isna().values.any() == True):
        data = data.dropna()
        
    X = data[["volatile acidity","chlorides","free sulfur dioxide","total sulfur dioxide","density",
               "pH","sulphates","alcohol"]]
    y = data [["quality"]]

    for i in range (len(X[["free sulfur dioxide"]])):
        if X[["free sulfur dioxide"]].values[i] == 'High':
            X.at[i, "free sulfur dioxide"] = 3
        elif X[["free sulfur dioxide"]].values[i] == 'Medium':
            X.at[i, "free sulfur dioxide"] = 2
        elif X[["free sulfur dioxide"]].values[i] == 'Low':
            X.at[i, "free sulfur dioxide"] = 1
        else:
            X.at[i, "free sulfur dioxide"] = 0
    
    for i in range (len(X[["density"]])):
        if X[["density"]].values[i] == 'Very High':
            X.at[i, "density"] = 0
        elif X[["density"]].values[i] == 'High':
            X.at[i, "density"] = 3
        elif X[["density"]].values[i] == 'Medium':
            X.at[i, "density"] = 2
        elif X[["density"]].values[i] == 'Low':
            X.at[i, "density"] = 1
            
    for i in range (len(X[["pH"]])):
        if X[["pH"]].values[i] == 'Very Basic':
            X.at[i, "pH"] = 3
        elif X[["pH"]].values[i] == 'Normal':
            X.at[i, "pH"] = 2
        elif X[["pH"]].values[i] == 'Very Acidic':
            X.at[i, "pH"] = 1
        else:
            X.at[i, "pH"] = 0
            
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y)
    
    return X, y


# In[3]:


inputDataset, outputDataset = loadData()


# In[4]:


# Normalized
inputDataset = preprocessing.normalize(inputDataset)


# In[5]:


# PCA
inputDataset = PCA(n_components=4).fit_transform(inputDataset)
print(inputDataset)
print(outputDataset)


# In[6]:


layers = {
    'input': 4,
    'hidden': 0,
    'output': 5
}

weights = {
    'input_to_hidden' : tf.Variable(tf.random_normal([layers['input'], layers['hidden']])),
    'hidden_to_output' : tf.Variable(tf.random_normal([layers['hidden'], layers['output']]))
}

biases = {
    'input_to_hidden' : tf.Variable(tf.random_normal([layers['hidden']])),
    'hidden_to_output' : tf.Variable(tf.random_normal([layers['output']]))
}

inputPlaceholder = tf.placeholder(tf.float32, [None, layers['input']])
targetPlaceholder = tf.placeholder(tf.float32, [None, layers['output']])


# In[7]:


def feedForward(datas):
    input_to_hidden_bias = tf.matmul(datas, weights['input_to_hidden']) + biases['input_to_hidden']
    activated_input_to_hidden = tf.nn.sigmoid(input_to_hidden_bias)

    hidden_to_output_bias = tf.matmul(activated_input_to_hidden, weights['hidden_to_output']) + biases['hidden_to_output']
    activated_hidden_to_output = tf.nn.sigmoid(hidden_to_output_bias)

    return activated_hidden_to_output


# In[8]:


output = feedForward(inputPlaceholder)

epoch = 5000
alpha = 0.5

errors = tf.reduce_mean(0.5 * (targetPlaceholder - output) ** 2)

optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(errors)

inputTrain, inputTest, outputTrain, outputTest = train_test_split(inputDataset, outputDataset, test_size=0.1)
inputTrain, inputValidationTest, outputTrain, outputValidationTest = train_test_split(inputTrain, outputTrain, test_size=0.2)


# In[9]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1, epoch + 1) :
        train_dict = {
            inputPlaceholder : inputTrain,
            targetPlaceholder : outputTrain
        }

        sess.run(train, feed_dict = train_dict)

        loss = sess.run(errors, feed_dict = train_dict)
        
        if i%100==0:
            print("Epoch : {} Error : {}".format(i, loss))
        if i%500==0:
                validation_dict = {
                    inputPlaceholder : inputValidationTest,
                    targetPlaceholder : outputValidationTest
                }

                sess.run(train, feed_dict = validation_dict)

                ValidationLoss = sess.run(errors, feed_dict = validation_dict)
                
                print("Validation Epoch : {} Error : {}".format(i, ValidationLoss))
                if i == 500:
                    lowestValidation = ValidationLoss
                if ValidationLoss < lowestValidation:
                    lowestValidation = ValidationLoss
                    f = open("validationTracking.txt","w")
                    f.write(str(ValidationLoss))
                    f.close()
                    

    matches = tf.equal(tf.argmax(targetPlaceholder, axis = 1), tf.argmax(output, axis = 1))
    accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))

    feed_test = {
        inputPlaceholder: inputTest,
        targetPlaceholder: outputTest
    }

    print("Accuracy: {}%".format(sess.run(accuracy, feed_dict = feed_test)*100 ))
    print("Lowest Validation Loss : {}".format(lowestValidation))
    


# In[ ]:




