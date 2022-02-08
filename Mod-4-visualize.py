#!/usr/bin/env python
# coding: utf-8

# In[5]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns # Seaborn is uses matplotlib and automates more complex plots
get_ipython().system('pip install plotly')
import plotly.express as px


# In[6]:


import numpy as np
import pandas as pd
np.random.seed(0) # change seed for different answer
n = 1000
tt = np.linspace(0,n-1,n)
x = np.random.rand(n)+tt/500
y = np.random.normal(0,x,n)
z = [0]
for i in range(1,n):
    z.append(min(max(-3,z[i-1]+y[i]*0.1),3)) #Create a time series that changes based on y[i]*0.1 staying within the range -3 to 3
data = pd.DataFrame(np.vstack((tt,x,y,z)).T,                    columns=['time','x','y','z'])
data['w'] = '0-499'
for i in range(int(n/2),n):
    data.at[i,'w'] = '500-999'
data.head()


# In[7]:


plt.plot(tt,z)
plt.show()


# In[8]:


plt.figure(1,figsize=(10,6))                         # adjust figure size
ax=plt.subplot(2,1,1)                                # subplot 1
plt.plot(tt,z,'r-',linewidth=3,label='z')            # plot red line
ax.grid()                                            # add grid
plt.ylabel('z'); plt.legend()                        # add ylabel, legend
plt.subplot(2,1,2)                                   # subplot 2
plt.plot(tt,x,'b.',label='x')                        # plot blue dots
plt.plot(tt,y,color='orange',label='y',alpha=0.7)    # plot orange line
plt.xlabel('time'); plt.legend()                      # labels
plt.savefig('04-myFig.png',transparent=True,dpi=600) # save figure
plt.show()                                           # show plot


# In[25]:


#Plot Activity
xt = np.array([0,0.1,0.2,0.3,0.5,0.8,1.0]);
yt = np.array([1.0,2.1,3.5,6.5,7.2,5.9,6.3]);
zt = xt*yt

plt.plot(xt,yt,'s-')
plt.plot(xt,zt,'^-')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[22]:


#Scatter Plot
#matplotlib
plt.scatter(x,y)
plt.show()


# In[13]:


#plotly
fig = px.scatter(data,x='x',y='y',color='w',size='x',hover_data=['w'])
fig.show()


# In[14]:


#Bar Chart
bins = np.linspace(-3,3,31)
plt.hist(y,bins,label='y')
plt.hist(z,bins,alpha=0.7,label='z')
plt.legend()
plt.show()


# In[26]:


#Bar Plot Activity
bins = np.linspace(-3,3,31)
nt = 1000
xt = np.random.rand(nt)
yt = np.random.normal(0,1,nt)
zt = xt*yt
plt.hist(yt,bins,label='yt')
plt.hist(zt,bins,alpha=0.7,label='zt')
plt.hist(xt,bins,alpha=0.7,label='xt')

plt.legend()
plt.show()


# In[27]:


sns.pairplot(data[['x','y','z','w']],hue=('w')) #hue shows a different color
plt.show()


# In[34]:


#Pair Plot Activity
nt = 100
xt = np.random.rand(nt)
yt = np.random.normal(0,1,nt)
zt = xt*yt
dt = pd.DataFrame(np.column_stack([xt,yt,zt]),columns=['xt','yt','zt'])
dt['Dist'] = 'First'
for i in range(int(nt/2),nt):
    dt.at[i,'Dist'] = 'Second'

sns.pairplot(dt[['xt','yt','zt','Dist']],hue=('Dist')) 


# In[38]:


#Box Plot
sns.boxplot(x='w',y='x',data=data)
plt.show()


# In[42]:


sns.boxplot(x='Dist',y='yt',data=dt)
plt.show()


# In[40]:


#Violin plot
sns.violinplot(x='w',y='x',data=data,size=6)
plt.show()


# In[43]:


#Violin Plot Exercise
sns.violinplot(x='Dist',y='zt',data=dt,size=6)
plt.show()


# In[50]:


#Joint plot
sns.jointplot('x','z',data=data,kind="hex")
plt.show()


# In[52]:


#Joint plot Activity
sns.jointplot('yt','zt',data=dt,kind="hex")
plt.show()


# In[53]:


#TCLab Activity
import tclab, time, csv
import numpy as np
try:
    n = 120 
    with open('04-tclab.csv',mode='w',newline='') as f:
        cw = csv.writer(f)
        cw.writerow(['Time','Q1','Q2','T1','T2'])
        with tclab.TCLab() as lab:
            print('t Q1 Q2 T1    T2')
            for t in range(n):
                if t%30==0:
                    Q1 = np.random.randint(0,81)
                    Q2 = np.random.randint(0,81)
                    lab.Q1(Q1); lab.Q2(Q2)
                cw.writerow([t,Q1,Q2,lab.T1,lab.T2])
                if t%5==0:
                    print(t,Q1,Q2,lab.T1,lab.T2)
                time.sleep(1)
    data4=pd.read_csv('04-tclab.csv')
except:
    print('Connect TCLab to generate data')
    url = 'http://apmonitor.com/do/uploads/Main/tclab_dyn_data2.txt'
    data4=pd.read_csv(url)
    data4.columns = ['Time','Q1','Q2','T1','T2']
    
data4.head()    


# In[68]:


plt.figure(figsize=(12,5))
plt.subplot(2,1,1)
plt.plot(data4['Time'],data4['Q1'],'r--', label='Q1')
plt.plot(data4['Time'],data4['Q2'],'g^', label='Q2')
plt.subplot(2,1,2)
plt.plot(data4['Time'],data4['T1'],'r--', label='T1')
plt.plot(data4['Time'],data4['T2'],'g^', label='T2')
#Pair plot

d4a= data4[['Q1','T1']].copy()
d4b= data4[['Q2','T2' ]].copy()

d4a.columns=['Q','T'] #replaces Q1 T1
d4b.columns=['Q','T']

d4a['Combo']='Heater 1'
d4b['Combo']='Heater 2'

d4=pd.concat([d4a,d4b],ignore_index=False) #stack on top of each other
sns.pairplot(d4,('Combo'))
plt.show()


# In[ ]:




