#!/usr/bin/env python
# coding: utf-8

# In[7]:


#Generate data
import numpy as np
np.random.seed(0)
n = 1000
time = np.linspace(0,n-1,n) 
x = np.random.rand(n) #uniform distribution between 0 and 1
y = np.random.normal(1,1,n) #normal distribution
data = np.vstack((time,x,y)).T #Combine time, x, and y with a vertical stack np.vstack and transpose .T for column oriented data.
np.savetxt('03-data.csv',data,header='time,x,y',delimiter=',',comments='')


# In[8]:


#Display data Distributions
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(x,10,label='x')
plt.hist(y,60,label='y',alpha=0.7)
plt.ylabel('Count'); plt.legend()
plt.show()


# In[9]:


#Data Analysis with numpy
import numpy as np
data = np.loadtxt('03-data.csv',delimiter=',',skiprows=1)

print('Dimension (rows,columns):')
print(np.size(data,0),np.size(data,1))

print('Average:')
print(np.mean(data,axis=0))

print('Standard Deviation:')
print(np.std(data,0)) #can omit the need to put axis

print('Median:')
print(np.median(data,0))


# In[11]:


#Analyze data exercise

import scipy
from scipy.stats import skew

pm=x*y
print('Mean:')
print(np.mean(pm))

print('Standard Deviation:')
print(np.std(pm))

print('Median:')
print(np.median(pm))

plt.hist(pm,30, label='pm')
plt.show()

print('Skewedness:')
print(scipy.stats.skew(pm))


# In[12]:


#Data analysis with pandas
import pandas as pd

data = pd.read_csv('03-data.csv')
data.describe()


# In[16]:


#Data analysis with pandas-profiling
try:
    import pandas as pd
    from pandas_profiling import ProfileReport
    import os
except:
    get_ipython().system('pip install --user pandas-profiling')
    get_ipython().system('jupyter nbextension enable --py widgetsnbextension')
    print('Restart the Kernel before proceeding')
    
# import data
url='http://apmonitor.com/pdc/uploads/Main/tclab_data2.txt'
data = pd.read_csv(url)
profile = ProfileReport(data, explorative=True, minimal=False)
#The profile report can be saved as an interactive web-page. 
#The web-page is saved to the current working directory that is displayed with os.getcwd().
profile.to_file('report.html')
print('File report.html saved to '+os.getcwd())


# In[17]:


#Generate a file from the TCLab data with seconds (t), heater levels (Q1 and Q2), and temperatures (lab.T1 and lab.T2).
#Record data every second for 120 seconds and change the heater levels every 30 seconds to a random number between 0 and 80
import tclab, time, csv
import pandas as pd
import numpy as np
try:
    # connect to TCLab if available
    n = 120 
    with open('03-tclab1.csv',mode='w',newline='') as f:
        cw = csv.writer(f)
        cw.writerow(['Time','Q1','Q2','T1','T2'])
        with tclab.TCLab() as lab:
            print('t Q1 Q2 T1    T2')
            for t in range(n):
                if t%30==0: #every 30 seconds Q1 and Q2 changes. If the remainder of t/30 =/=0, Qs dont change
                    Q1 = np.random.randint(0,81)
                    Q2 = np.random.randint(0,81)
                    lab.Q1(Q1); lab.Q2(Q2)
                cw.writerow([t,Q1,Q2,lab.T1,lab.T2])
                if t%5==0: #print every 5 seconds
                    print(t,Q1,Q2,lab.T1,lab.T2)
                time.sleep(1)
    file = '03-tclab1.csv'
    data1=pd.read_csv(file)
except:
    print('No TCLab device found, reading online file')
    url = 'http://apmonitor.com/do/uploads/Main/tclab_dyn_data2.txt'
    data1=pd.read_csv(url)


# In[18]:


#Use requests to download a sample TCLab data file for the analysis.
import requests
import os
url = 'http://apmonitor.com/pdc/uploads/Main/tclab_data2.txt'
r = requests.get(url)
with open('03-tclab2.csv', 'wb') as f:
    f.write(r.content)
    
print('File 03-tclab2.csv retrieved to current working directory: ')
print(os.getcwd())


# In[24]:


import pandas as pd

data1 = pd.read_csv('03-tclab1.csv')
data2 = pd.read_csv('03-tclab2.csv')

data1.describe();
data2.describe();

Diff = abs(data1.describe()-data2.describe());

print('Statistics on tclab1')
print(data1.describe())

print('Statistics on tclab2')
print(data2.describe())

print('Differences:')
print(Diff)

print(Diff.head(3))


# In[25]:


profile = ProfileReport(data1, explorative=True, minimal=False)
#The profile report can be saved as an interactive web-page. 
#The web-page is saved to the current working directory that is displayed with os.getcwd().
profile.to_file('report.html')
print('File report.html saved to '+os.getcwd())


# In[ ]:




