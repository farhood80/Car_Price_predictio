#!/usr/bin/env python
# coding: utf-8

# <b> Car Price Prediction Project
#     

# In[53]:


#importing the depedencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics


# In[54]:


#Data Processing and Collection

#loading Data as pandas dataframe

Car_price = pd.read_csv('/home/farhood/Desktop/car data.csv')

#first 10 row of the dataset 
Car_price.head(10)

#Note : you can change the number of the appering rows by changing the n in the 'head(n)'


# In[55]:


# how to know all rows and columns in the dataset 
Car_price.shape
    


# In[56]:


#get more informations about the dataset
Car_price.info()

#for describing the dataset
Car_price.describe()


# In[57]:


# check the number of the missing values

Car_price.isnull().sum() 


# In[58]:


# check how many cars are petrol or diesel

print(Car_price.Fuel_Type.value_counts())
print(Car_price.Seller_Type.value_counts())
print(Car_price.Transmission.value_counts())


# In[59]:


#Encoding The Categorical Data

# encoding "Fuel_Type" Column
Car_price.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)

# encoding "Seller_Type" Column
Car_price.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)

# encoding "Transmission" Column
Car_price.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)


# In[60]:


Car_price.head(10)


# In[61]:


#Spliting Data and target

#
X = Car_price.drop(['Car_Name', 'Selling_Price'], axis =1)
Y = Car_price['Selling_Price']


# In[62]:


# Spliting Data into Train and Test

x_train , x_test, y_train , y_test = train_test_split(X,Y, test_size = 0.1, random_state = 2) 


# In[63]:


# Model training

#loading the linear regression model

lin_reg_model = LinearRegression()


# In[64]:


lin_reg_model.fit(x_train,y_train)


# In[65]:


# Model Evalution

#prediction on Training data

training_data_prediction = lin_reg_model.predict(x_train)


# In[66]:


# compare the x_trian result with y_train

#R squared Error

error_score = metrics.r2_score(y_train, training_data_prediction)

print("the R error square is equal : ", error_score)


# In[67]:


#visual the actual and predcit prices
plt.scatter(y_train , training_data_prediction, color = "green")
plt.xlabel("Actual prices")
plt.ylabel("predicted prices")
plt.title("comparesion of the Actual and predcited prices")
plt.show()


# In[68]:


#prediction on Test data

test_data_prediction = lin_reg_model.predict(x_test)


# In[69]:


# compare the x_test result with y_test

#R squared Error

error_score = metrics.r2_score(y_test, test_data_prediction)

print("the R error square is equal : ", error_score)


# In[70]:


#visual the actual and predcit prices
plt.scatter(y_test , test_data_prediction, color = "green")
plt.xlabel("Actual prices")
plt.ylabel("predicted prices")
plt.title("comparesion of the Actual and predcited prices")
plt.show()


# In[71]:


#Lasso regression

# Model training

#loading the linear regression model

lass_reg_model = Lasso()


# In[72]:


lass_reg_model.fit(x_train,y_train)


# In[73]:


# Model Evalution

#prediction on Training data

training_data_prediction = lass_reg_model.predict(x_train)


# In[74]:


# compare the x_trian result with y_train

#R squared Error

error_score = metrics.r2_score(y_train, training_data_prediction)

print("the R error square is equal : ", error_score)


# In[75]:


#visual the actual and predcit prices
plt.scatter(y_train , training_data_prediction, color = "green")
plt.xlabel("Actual prices")
plt.ylabel("predicted prices")
plt.title("comparesion of the Actual and predcited prices")
plt.show()


# In[76]:


#prediction on Test data

test_data_prediction = lass_reg_model.predict(x_test)


# In[77]:


# compare the x_test result with y_test

#R squared Error

error_score1 = metrics.r2_score(y_test, test_data_prediction)

print("the R error square is equal : ", error_score1)


# In[78]:


# compare the x_test result with y_test

#R squared Error

error_score = metrics.r2_score(y_test, test_data_prediction)

print("the R error square is equal : ", error_score)


# In[79]:


#visual the actual and predcit prices
plt.scatter(y_test , test_data_prediction, color = "green")
plt.xlabel("Actual prices")
plt.ylabel("predicted prices")
plt.title("comparesion of the Actual and predcited prices")
plt.show()

