#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


df= pd.read_csv(r"C:\Users\nitin\Downloads\archive\Iris.csv")
df


# In[6]:


df.describe()


# In[7]:


df["Species"].unique()


# In[8]:


df.groupby("Species").size()


# In[9]:


corr = df.corr()
plt.subplots(figsize=(10,16))
sns.heatmap(corr, annot=True)


# In[10]:


x = df.iloc[:, 1:5].values
y = df.iloc[:, 5].values


# In[11]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# In[12]:


y


# In[13]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


# In[14]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[16]:


plt.figure()
sns.pairplot(df.drop("Id", axis=1), hue = "Species", height = 3, markers=["o", "s", "D"])
plt.show()


# In[21]:


from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn.model_selection import cross_val_score


# In[23]:


# Instantiate learning model (k=3)
classifier = KNeighborsClassifier(n_neighbors=3) 


# In[24]:


# Fittting the model 
classifier.fit(x_train, y_train)


# In[25]:


# Predicting the Test set results 
y_pred = classifier.predict(x_test)


# In[27]:


accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy:' + str(round(accuracy, 2)) + '%')+

