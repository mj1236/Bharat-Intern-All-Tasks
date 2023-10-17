#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.linear_model import LinearRegression


# In[2]:


df = pd.read_csv('winequality-red.csv')
print(df.head())


# In[3]:


df.info()


# In[4]:


df.describe().T


# In[5]:


df.isnull().sum()


# In[6]:


for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

df.isnull().sum().sum()


# In[7]:


df.hist(bins=20, figsize=(10, 10))
plt.show()


# In[8]:


plt.bar(df['quality'], df['alcohol'])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()


# In[9]:


plt.figure(figsize=(12, 12))
sb.heatmap(df.corr() > 0.7, annot=True, cbar=False)
plt.show()


# In[10]:


df = df.drop('total sulfur dioxide', axis=1)


# In[11]:


df['best quality'] = [1 if x > 5 else 0 for x in df.quality]


# In[12]:


df.replace({'white': 1, 'red': 0}, inplace=True)


# In[13]:


features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']

xtrain, xtest, ytrain, ytest = train_test_split(
	features, target, test_size=0.2, random_state=40)

xtrain.shape, xtest.shape


# In[14]:


norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)


# In[15]:


lm = LinearRegression()
lm.fit(xtrain,ytrain)


# In[16]:


print(lm.intercept_)
print(lm.coef_)


# In[17]:


predictions = lm.predict(xtest)


# In[27]:


models = [LinearRegression()]

for i in range(1):
	models[i].fit(xtrain, ytrain)

	print(f'{models[i]} : ')
	print('Training Accuracy : ', metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
	print('Validation Accuracy : ', metrics.roc_auc_score(
		ytest, models[i].predict(xtest)))
	print()

