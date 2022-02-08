#!/usr/bin/env python
# coding: utf-8

# In[1]:


#File open for read or write
# write a test file with a message
f = open('02-file.txt','w')
f.write('This is a test file')
f.close()

import os
print('File stored in: ' + os.getcwd())

# read and print the file contents
f = open('02-file.txt')
print(f.read())
f.close()


# In[2]:


#open and csv 
clist = ['x','y','z']
m = [[1,2,3],     [4,5,6],     [7,8,9]]

import csv
with open('02-data1.csv',mode='w',newline='') as f:
    cw = csv.writer(f)
    cw.writerow(clist)
    for i in range(len(m)):
        cw.writerow(m[i])


# In[3]:


#numpy write CSV
import numpy as np
np.savetxt('02-data2.csv',m,delimiter=',',comments='',header='x,y,z')


# In[4]:


#pandas writes CSV
import pandas as pd
df = pd.DataFrame(m,columns=clist)
df.to_csv('02-data3.csv',index=False)


# In[6]:


#pandas writes XLSX and JSON

df.to_json('02-data3.json',orient='table',index=False)
df.to_excel('02-data3.xlsx',index=False)


# In[20]:


#Use numpy to create 51 equally spaced values for x between 0 and 100. 
#Calculate y=x**2 and z=x**3 that are derived from x. 
#Store x, y, and z in a CSV file with headings in file 02-test.csv.
import numpy as np
x=np.linspace(0,100,51);
y=x**2;
z=x**3;
data=np.column_stack((x,y,z))
np.savetxt('02-test.csv',data,delimiter=',',comments='',header='x,y,z')


# In[22]:


#Use numpy to read CSV
data = np.loadtxt('02-data1.csv',delimiter=',',skiprows=1) #skiprows=1 to skip the header row. Numpy does not label the rows or columns and only stores the CSV values.
print(data)


# In[24]:


#Use pandas to read CSV
data = pd.read_csv('02-data1.csv')
data.head() #The data.head() and data.tail() functions prints up to the first or last 5 values, respectively.
data.describe() #nice statistics


# In[25]:


#Use pandas to read the 02-test.csv file created above. Display the first 5 rows of the file.
data = pd.read_csv('02-test.csv')
data.head()


# In[33]:


#It is also possible to delete files using the os (operating system) module.
import os
import glob #The glob module builds a list of files that start with 02-data and end with .csv. It uses the wildcard character * to select any files that match the first and last parts.
filelist = glob.glob('02-data*.csv')

if filelist==[]:
    print('No files to delete')
    ans='no'
else:
    ans = input('Delete files '+str(filelist)+'? ')

if ans[0].lower()=='y':
    for f in filelist:
        os.remove(f)


# In[29]:


import os
import glob
filelist=glob.glob('02-test.csv')

if filelist==[]:
    print('No files to delete')
    ans='no'
else:
    ans = input('Delete files '+str(filelist)+'? ')

if ans[0].lower()=='y':
    for f in filelist:
        os.remove(f) #this line is enough to delete the file os.remove('02-test.csv')


# In[56]:


#Write data file 02-tclab.csv with 5 columns that include time in seconds (t), heater levels (Q1 and Q2), and temperatures (lab.T1 and lab.T2). 
#Include a data row every second for 20 seconds. 
#The starting script only prints those values to the screen but they also need to be saved to a file
import tclab
import time
import numpy as np
import pandas as pd
n = 20
Q1 = 30; Q2 = 70
data=np.zeros([n,5]); #initialization
clist=['t','Q1','Q2','T1','T2']
with tclab.TCLab() as lab:
    lab.Q1(Q1); lab.Q2(Q2)
    print('t Q1 Q2 T1    T2')
    for t in range(n):
        data[t]=[t, Q1, Q2, lab.T1,lab.T2] #writing each iteration into data matrix
        print(t,Q1,Q2,lab.T1,lab.T2)
        time.sleep(1)
df = pd.DataFrame(data,columns=clist)
df.to_csv('02-tclab.csv',index=False)
data = pd.read_csv('02-tclab.csv')
data.head() #print out first 5 rows


# In[2]:


#instructor solution
import tclab
import time
import csv
import pandas as pd
n = 20
Q1 = 30; Q2 = 70
with tclab.TCLab() as lab:
    lab.Q1(Q1); lab.Q2(Q2)
    with open('02-tclab.csv',mode='w',newline='') as f:
        cw=csv.writer(f)
        cw.writerow(['t','Q1','Q2','T1','T2'])
        for t in range(n):
            print(t,Q1,Q2,lab.T1,lab.T2)
            cw.writerow([t,Q1,Q2,lab.T1,lab.T2])
            time.sleep(1)

try:
    data=pd.read_csv('02-tclab.csv')
    print('read from file')
except:
    print('read from url')
    url='http://apmonitor.com/pdc/uploads/Main/tclab_data2.txt'
    data=pd.read_csv(url)
data.head()


# In[ ]:




