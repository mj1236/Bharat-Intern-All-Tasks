#!/usr/bin/env python
# coding: utf-8

# In[4]:


# DataFlair Iris Classification
# Import Packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels'] # As per the iris dataset information


# In[6]:


# Load the data
df = pd.read_csv('iris.data', names=columns)


# In[7]:


df.head()


# In[8]:


# Some basic statistical analysis about the data
df.describe()


# In[9]:


# Visualize the whole dataset
sns.pairplot(df, hue='Class_labels')


# In[11]:


# Seperate features and target  
data = df.values
X = data[:,0:4]
Y = data[:,4]


# In[30]:


# Calculate avarage of each features for all classes
Y_Data = np.array([np.average(X[:, i][Y==j].astype('float32')) for i in range (X.shape[1]) for j in (np.unique(Y))])
Y_Data_reshaped = Y_Data.reshape(4, 3)
Y_Data_reshaped = np.swapaxes(Y_Data_reshaped, 0, 1)
X_axis = np.arange(len(columns)-1)
width = 0.25


# In[31]:


# Plot the avarage
plt.bar(X_axis, Y_Data_reshaped[0], width, label = 'Setosa')
plt.bar(X_axis+width, Y_Data_reshaped[1], width, label = 'Versicolour')
plt.bar(X_axis+width*2, Y_Data_reshaped[2], width, label = 'Virginica')
plt.xticks(X_axis, columns[:4])
plt.xlabel("Features")
plt.ylabel("Value in cm.")
plt.legend(bbox_to_anchor=(1.3,1))
plt.show()


# In[34]:


# Split the data to train and test dataset.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# In[35]:


# Support vector machine algorithm
from sklearn.svm import SVC
svn = SVC()
svn.fit(X_train, y_train)


# In[36]:


# Predict from the test dataset
predictions = svn.predict(X_test)


# In[37]:


# Calculate the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)


# In[38]:


# A detailed classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


# In[39]:


X_new = np.array([[3, 2, 1, 0.2], [  4.9, 2.2, 3.8, 1.1 ], [  5.3, 2.5, 4.6, 1.9 ]])
#Prediction of the species from the input vector
prediction = svn.predict(X_new)
print("Prediction of Species: {}".format(prediction))


# In[40]:


# Save the model
import pickle
with open('SVM.pickle', 'wb') as f:
    pickle.dump(svn, f)


# In[41]:


# Load the model
with open('SVM.pickle', 'rb') as f:
    model = pickle.load(f)


# In[42]:


model.predict(X_new)


# In[ ]:




