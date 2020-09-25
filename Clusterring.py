#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


# In[2]:


def loadData():
    data = pd.read_csv('E202-COMP7117-TD01-00 - clustering.csv')
    
    if (data.isna().values.any() == True):
        data = data.dropna()
        
    X = data[["ProductRelated_Duration","ExitRates","SpecialDay","VisitorType","Weekend"]]

    for i in range (len(X[["SpecialDay"]])):
        if X[["SpecialDay"]].values[i] == 'HIGH':
            X.at[i, "SpecialDay"] = 2
        elif X[["SpecialDay"]].values[i] == 'NORMAL':
            X.at[i, "SpecialDay"] = 1
        elif X[["SpecialDay"]].values[i] == 'LOW':
            X.at[i, "SpecialDay"] = 0
    
    for i in range (len(X[["VisitorType"]])):
        if X[["VisitorType"]].values[i] == 'Returning_Visitor':
            X.at[i, "VisitorType"] = 2
        elif X[["VisitorType"]].values[i] == 'New_Visitor':
            X.at[i, "VisitorType"] = 1
        elif X[["VisitorType"]].values[i] == 'Other':
            X.at[i, "VisitorType"] = 0
            
    X['Weekend'] = X['Weekend'] * 1
    
    print(X)
    print(X.dtypes)
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    # PCA
    X = PCA(n_components=3).fit_transform(X)
    
    return X


# In[3]:


dataset = loadData()
print(dataset)


# In[4]:


class SOM: 
    def __init__(self, height, width, input_dimension):
        self.height = height
        self.width = width
        self.input_dimension = input_dimension

        #row = cluster amount
        #column = input dimension
        self.weight = tf.Variable(tf.random_normal([width*height, input_dimension]))
        self.input = tf.placeholder(tf.float32, [input_dimension])

        self.location = [tf.to_float([y,x]) for y in range(height) for x in range(width)]
        
        self.bmu = self.getBMU()

        self.update_weight = self.update_neighbour()


    def getBMU(self):
        #Eucledian distance
        square_distance = tf.square(self.input - self.weight)
        distance = tf.sqrt(tf.reduce_sum(square_distance, axis=1))

        #Get BMU index
        bmu_index = tf.argmin(distance)
        #Get the position
        bmu_position = tf.to_float([tf.div(bmu_index,self.width), tf.mod(bmu_index, self.width)])
        return bmu_position


    def update_neighbour(self):
        
        learning_rate = 0.1

        #Formula calculate sigma / radius
        sigma = tf.to_float(tf.maximum(self.width, self.height) / 2)

        #Eucledian Distance between BMU and location
        square_difference = tf.square(self.bmu - self.location)
        distance = tf.sqrt(tf.reduce_sum(square_difference,axis=1))

        #Calculate Neighbour Strength based on formula
        # NS = tf.exp((- distance ** 2) /  (2 * sigma ** 2))
        NS = tf.exp(tf.div(tf.negative(tf.square(distance)), 2 * tf.square(sigma)))

        #Calculate rate before reshape
        rate = NS * learning_rate

        #Reshape to [width * height, input_dimension]
        rate_stacked = tf.stack([tf.tile(tf.slice(rate,[i],[1]), [self.input_dimension]) 
            for i in range(self.width * self.height)])

        #Calculate New Weight
        new_weight = self.weight + rate_stacked * (self.input - self.weight)

        return tf.assign(self.weight, new_weight)


    def train(self, dataset, num_of_epoch):
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            for i in range(num_of_epoch+1):
                for data in dataset:
                    dictionary = {
                        self.input : data
                    }
                    sess.run(self.update_weight, feed_dict=dictionary)

            location = sess.run(self.location)
            weight = sess.run(self.weight)
            cluster = [[] for i in range(self.height)]

            for i, loc in enumerate(location):
                print(i,loc[0])
                cluster[int(loc[0])].append(weight[i])

            self.cluster = cluster


# In[ ]:


input_dimension = 3

som = SOM(3,3,input_dimension)

som.train(dataset, 5000)

plt.imshow(som.cluster)
plt.show()

