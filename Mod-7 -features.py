#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import google stock data in 2017
import pandas as pd
import numpy as np
url = 'http://apmonitor.com/che263/uploads/Main/goog.csv'
data = pd.read_csv(url)
data = data.drop(columns=['Adj Close'])
data.head()


# In[5]:


features = ['Open','Volatility','Change','Volume']
data['Volatility'] = (data['High']-data['Low']).diff()
data['Change'] = (data['Close']-data['Open']).diff()
# any other features?
data.head()


# In[6]:


data['Close_diff'] = np.roll(data['Close'].diff(),-1) #shifts all the values up by one to indicate the change for the next day 
data=data.dropna()
label = ['Buy/Sell']
data['Buy/Sell'] = np.sign(data['Close_diff'])
data.head()


# In[ ]:




