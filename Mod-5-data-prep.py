#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Generate Sample Data
#corrupted with NaN and outlier
import numpy as np
import pandas as pd
np.random.seed(1)
n = 100
tt = np.linspace(0,n-1,n)
x = np.random.rand(n)+10+np.sqrt(tt)
y = np.random.normal(10,x*0.01,n)
x[1] = np.nan; y[2] = np.nan  # 2 NaN (not a number)
for i in range(3):            # add 3 outliers (bad data)
    ri = np.random.randint(0,n)
    x[ri] += np.random.rand()*100
data = pd.DataFrame(np.vstack((tt,x,y)).T,                    columns=['time','x','y'])
data.head()


# In[4]:


#Visualize data
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.semilogy(tt,x,'r.',label='x')
plt.semilogy(tt,y,'b.',label='y')
plt.legend(); plt.xlabel('time')
plt.text(50,60,'Outliers')
plt.show()


# In[3]:


z = np.array([[      1,      2],
              [ np.nan,      3],
              [      4, np.nan],
              [      5,      6]])
iz = np.any(np.isnan(z), axis=1) #look for anything that is not a number
print(~iz)
z = z[~iz]
print(z)


# In[4]:


# drop any row with bad (NaN) values
data = data.dropna()
data.head()


# In[5]:


#Use histogram to identifier outlier
plt.boxplot(data['x'])
plt.show()


# In[6]:



data = data[data['x']<30] #keep x<30
plt.boxplot(data['x'])
plt.show()


# In[5]:


#Time activity #Count your MISSISSIPPI!
import time
from IPython.display import clear_output
tsec = []
input('Press "Enter" to record 1 second intervals'); t = time.time()
for i in range(10):
    clear_output(); input('Press "Enter": ' + str(i+1))
    tsec.append(time.time()-t); t = time.time()
clear_output(); print('Completed. Add boxplot to identify outliers')


# In[14]:


plt.boxplot(tsec)
plt.show()


# In[6]:


#Scale Data
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
ds = s.fit_transform(data) #ds is returned as a numpy; can convert back to pandas
print(ds[0:5]) # print 5 rows


# In[16]:


ds = pd.DataFrame(ds,columns=data.columns)
ds.head()
#Converting back to pandas


# In[17]:


#Divide the data into train and test sets
divide = int(len(ds)*0.8)
train = ds[0:divide]
test = ds[divide:]
print(len(train),len(test))


# In[42]:


#TCLab Activity
import tclab, time, csv
import numpy as np

try:
    with tclab.TCLab() as lab:
        with open('05-tclab.csv',mode='w',newline='') as f:
            cw = csv.writer(f)
            cw.writerow(['Time','Q1','Q2','T1','T2'])
            print('t Q1 Q2 T1    T2')
            for t in range(180):
                T1 = lab.T1; T2 = lab.T2
                # insert bad values
                bad = np.random.randint(0,30)
                T1=np.nan if bad==10 else T1
                T2=np.nan if bad==15 else T2
                # insert random number (possibly outlier)
                outlier = np.random.randint(-40,150)
                T1=outlier if bad==20 else T1
                T2=outlier if bad==25 else T2
                # change heater
                if t%30==0:
                    Q1 = np.random.randint(0,81)
                    Q2 = np.random.randint(0,81)
                    lab.Q1(Q1); lab.Q2(Q2)
                cw.writerow([t,Q1,Q2,T1,T2])
                if t%10==0:
                    print(t,Q1,Q2,T1,T2)
                time.sleep(1)
            data5=pd.read_csv('05-tclab.csv')
except:
    print('Connect TCLab to generate new data')
    print('Importing data from online source')
    url = 'http://apmonitor.com/do/uploads/Main/tclab_bad_data.txt'
    data5=pd.read_csv(url)


# In[7]:


print('Importing data from online source')
url = 'http://apmonitor.com/do/uploads/Main/tclab_bad_data.txt'
data5=pd.read_csv(url)


# In[8]:


data5=data5[['T1','T2']] #Keeping temperature
data5=data5[data5.diff().abs()<5] #keeping any value where temp diff is less than 5
data5=data5.dropna() #drop any NaN
data5.plot()


# In[9]:


from sklearn.preprocessing import StandardScaler
s=StandardScaler()
ds5=s.fit_transform(data5)
divide5 = int(len(ds5)*0.8)
train5 = ds5[0:divide5]
test5 = ds5[divide5:]
print(len(train5),len(test5))


# In[12]:


#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
#train test split from scikit
from sklearn.model_selection import train_test_split
train,test = train_test_split(ds5,test_size=0.8,shuffle=True)
print(len(train),len(test))

