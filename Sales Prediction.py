#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
import seaborn as sb


# In[3]:


df=pd.read_csv(r"N:\TE\2nd Sem\Sales Prediction\Advertising.csv")
df


# In[4]:


df=df.drop(columns=["Unnamed: 0"])


# In[5]:


df.head()


# In[7]:


df.shape


# In[9]:


x=df.iloc[:,0:-1]
x


# In[10]:


y=df.iloc[:,-1]
y


# In[11]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=43)


# In[12]:


x_train


# In[13]:


x_test


# In[14]:


y_train


# In[15]:


y_test


# In[16]:


x_train=x_train.astype(int)
y_train=y_train.astype(int)
x_test=x_train.astype(int)
y_test=y_train.astype(int)


# In[19]:


from sklearn.preprocessing import StandardScaler
Sc=StandardScaler()
x_train_scaled=Sc.fit_transform(x_train)


# In[20]:


x_test_scaled=Sc.fit_transform(x_test)


# In[21]:


from sklearn.linear_model import LinearRegression


# In[22]:


lr=LinearRegression()


# In[23]:


lr.fit(x_train_scaled,y_train)


# In[27]:


y_pred=lr.predict(x_test_scaled)


# In[28]:


from sklearn.metrics import r2_score


# In[29]:


r2_score(y_test,y_pred)


# In[30]:


import matplotlib.pyplot as plt


# In[31]:


plt.scatter(y_test,y_pred,c='g')

