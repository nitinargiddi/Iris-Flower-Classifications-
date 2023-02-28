#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
import seaborn as sb


# In[4]:


df=pd.read_csv(r"N:\TE\2nd Sem\archive (2)\spam.csv")
df


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


x=df.v2
x


# In[8]:


y=df.v1
y


# In[9]:


y.replace(to_replace='ham',value=1,inplace=True)
y.replace(to_replace='spam',value=0,inplace=True)
y


# In[10]:


y=df.v1
y.value_counts()


# In[11]:


sb.histplot(y)


# In[15]:


from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,random_state=25)


# In[16]:


gok =pd.concat([x_train, y_train], axis=1)
# separate minority & majority classes 
spam = gok[gok.v1==0]
ham = gok[gok.v1==1]

