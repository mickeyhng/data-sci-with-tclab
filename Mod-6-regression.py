#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
n = 31
x = np.linspace(0,3,n)
z = np.array([1.89,-0.12,-0.32,2.11,-0.25,1.01,0.17,2.75,2.01,5.5,     0.87,3.31,2.29,2.73,3.67,3.92,4.29,5.27,3.85,4.26,     5.75,5.82,6.36,9.13,7.61,9.52,9.53,12.49,12.29,13.7,14.12])
data = pd.DataFrame(np.vstack((x,z)).T,columns=['x','z'])
plt.plot(x,z,'ro',label='Measured')
plt.xlabel('x'); plt.ylabel('z'); plt.legend()
plt.show()


# In[ ]:


#Linear Regression with numpy
p1 = np.polyfit(x,z,1)

print('Slope, Intercept:' + str(p1))

plt.plot(x,z,'ro',label='Measured (z)')
plt.plot(x,np.polyval(p1,x),'b-',label='Predicted (y)')
plt.legend(); plt.show()


# In[ ]:


#R2 value
from sklearn.metrics import r2_score
meas  = [3.0, 2.0, 1.9, 7.1]
model = [2.5, 1.8, 2.0, 8.0]
r2_score(meas, model)


# In[ ]:


#statsmodels perrforms OLS with summary; x is augumented with a np.ones(n) to predict the intercept
import statsmodels.api as sm
xc = sm.add_constant(x) #more conveninent over xc=np.vstack((x,np.ones(n))).T
model = sm.OLS(z,xc).fit()
predictions = model.predict(xc)
model.summary()


# In[ ]:


#Linear Regression Activity
xr = [0.0,1.0,2.0,3.5,5.0]
yr = [0.7,0.55,0.34,0.3,0.2]

p2 = np.polyfit(xr,yr,1)
print('Slope, Intercept:' + str(p1))
plt.plot(xr,yr,'ro',label='Measured')
plt.plot(xr,np.polyval(p2,xr),'b-',label='Predicted (y)')
plt.xlabel('x'); plt.ylabel('y'); plt.legend()

#Polyval
r2squared=r2_score(yr,np.polyval(p2,xr))
print('Rsquared value:' +str(r2squared))

#OLS
xc = sm.add_constant(xr)
model = sm.OLS(yr,xc).fit()
predictions = model.predict(xc)
model.summary()


# In[ ]:


#Polynomial Regression
p2 = np.polyfit(x,z,2)
print(p2)
plt.plot(x,z,'ro',label='Measured (z)')
plt.plot(x,np.polyval(p2,x),'b-',label='Predicted (y)')
plt.legend(); plt.show()


# In[ ]:


import statsmodels.api as sm
xc = np.vstack((x**2,x,np.ones(n))).T
model = sm.OLS(z,xc).fit()
predictions = model.predict(xc)
model.summary()


# In[ ]:


#Polynomial Regression Activity
xr = [0.0,1.0,2.0,3.5,5.0]
yr = [1.7,1.45,1.05,0.4,0.2]
p2 = np.polyfit(xr,yr,3)
xsmooth=np.linspace(0.1,5,50)#smoothing out the polynomial curve
plt.plot(xr,yr,'ro',label='Measured (z)')
plt.plot(xsmooth,np.polyval(p2,xsmooth),'b-',label='Predicted (y)')
plt.legend(); plt.show()


# In[ ]:


#Nonlinear Regression
from scipy.optimize import curve_fit
def f(x,a,b,c):
    return a * np.exp(b*x)+c
p, pcov = curve_fit(f,x,z)
print('p = '+str(p))
plt.plot(x,z,'ro')
plt.plot(x,f(x,*p),'b-') #asterik avoids putting a b c individually
plt.show()


# In[ ]:


#Nonlinear regression with natural log model
import math
xr = np.array([0.1,1.0,2.0,3.5,5.0])
yr = np.array([0.2,0.4,1.05,1.45,1.7])
def f(x,a,b):
    return a * np.log(b*x)
p, pcov = curve_fit(f,xr,yr, maxfev=2000, p0=[0.1,5])#add max iterations and initial guess
print('p = '+str(p))
plt.plot(xr,yr,'ro')
plt.plot(xsmooth,f(xsmooth,*p),'b-') #Smoothing the curve
plt.show()


# In[ ]:


#Machine learning
get_ipython().run_line_magic('matplotlib', 'inline')
#make plot interactive use %matplotlib notebook
from mpl_toolkits.mplot3d import Axes3D

import math
def f(x,y):
    return 2*math.cos(x)*y + x*math.cos(y) - 3*x*y

n = 500
x = (np.random.rand(n)-0.5)*2.0
y = (np.random.rand(n)-0.5)*2.0
z = np.empty_like(x)
for i in range(n):
    z[i] = f(x[i],y[i])
data = pd.DataFrame(np.vstack((x,y,z)).T,columns=['x','y','z'])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z,c=z,cmap='plasma')
plt.show()


# In[ ]:


#scaling
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
ds = s.fit_transform(data)
ds = pd.DataFrame(ds,columns=data.columns)


# In[ ]:


# no data scaling
ds = data

# data splitting into train and test sets
from sklearn.model_selection import train_test_split
train,test = train_test_split(ds, test_size=0.2, shuffle=True)


# In[ ]:


#Function for plotting
def fit(method):
    # create points for plotting surface
    xp = np.arange(-1, 1, 0.1)
    yp = np.arange(-1, 1, 0.1)
    XP, YP = np.meshgrid(xp, yp)

    model = method.fit(train[['x','y']],train['z'])
    zp = method.predict(np.vstack((XP.flatten(),YP.flatten())).T)
    ZP = zp.reshape(np.size(XP,0),np.size(XP,1))

    r2 = method.score(test[['x','y']],test['z'])
    print('R^2: ' + str(r2))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ds['x'],ds['y'],ds['z'],c=z,cmap='plasma',label='data')
    ax.plot_surface(XP, YP, ZP, cmap='coolwarm',alpha=0.7,
                    linewidth=0, antialiased=False)
    plt.show()
    return


# In[ ]:


#Linear Regression with sklearn
from sklearn import linear_model
lm = linear_model.LinearRegression()
fit(lm)


# In[ ]:


#K-Nearest Neighbors
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=2)
fit(knn)


# In[ ]:


#Support Vector Regressor
from sklearn import svm
s = svm.SVR(gamma='scale')
fit(s)


# In[ ]:


#Multilayer Perceptron Neural Network
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(hidden_layer_sizes=(3), 
                  activation='tanh', solver='lbfgs') #relu for activation
fit(nn)


# In[ ]:


#Regressor Activity

#Decision Tree Regressor
from sklearn import tree
dt = tree.DecisionTreeRegressor()
fit(dt)


# In[ ]:


#Passive Aggressive Regressor
from sklearn.linear_model import PassiveAggressiveRegressor
par = PassiveAggressiveRegressor(max_iter=10000,tol=1e-3)
fit(par)


# In[ ]:


#LARS 
from sklearn import linear_model
reg=linear_model.LassoLars(alpha=0.01, normalize=True)
fit(reg)


# In[ ]:


#Deep Learning with gekko
from gekko import brain
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0.0,2*np.pi)
y = np.sin(x)

b = brain.Brain(remote=False)
b.input_layer(1)
b.layer(linear=2)
b.layer(tanh=2)
b.layer(linear=2)
b.output_layer(1)
b.learn(x,y,disp=False)      

xp = np.linspace(-2*np.pi,4*np.pi,100)
yp = b.think(xp)  

plt.figure()
plt.plot(x,y,'bo',label='meas (label)')
plt.plot(xp,yp[0],'r-',label='model')
plt.legend(); plt.show()


# In[4]:


#TCLab Activity 
import tclab 
import time
import numpy as np
n = 60 
t=np.linspace(0,30,n+1)
T1=np.empty_like(t)
T2=np.empty_like(t)
with tclab.TCLab() as lab:
    lab.Q1(80);lab.Q2(60);
    for i in range(n+1):
        T1[i]=lab.T1
        T2[i]=lab.T2
        print(T1[i],T2[i])
        time.sleep(0.5)


# In[14]:


#Linear Regression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
p1 = np.polyfit(t,T2,1)

print('Slope, Intercept:' + str(p1))

plt.plot(t,T2,'bo',label='Measured T2')
plt.plot(t,np.polyval(p1,t),'k-',label='Predicted T2')
plt.legend(); plt.show()
print('R2 is ' + str( r2_score(T2,np.polyval(p1,t))))


# In[29]:


get_ipython().system('pip install tensorflow')


# In[34]:


#Nonlinear regression
#Exponential Curvefit
from scipy.optimize import curve_fit
from sklearn.neural_network import MLPRegressor
def f(x,a,b,c):
    return a + b * np.exp(c*x)
p, pcov = curve_fit(f,t,T1)
print('p = '+str(p))

#Neural networking
nn = MLPRegressor(hidden_layer_sizes=(3), activation='tanh', solver='lbfgs')
tr = t.reshape(-1,1) #let python determine the shape
model=nn.fit(tr,T1) #training
Tp=nn.predict(tr) #predicting
          
plt.plot(t,T1,'ro', label='Measured T1')
plt.plot(t,f(t,*p),'b-',label='Exponential Curve fit T1')
plt.plot(t,Tp, 'g^', label = 'Neural Networking T1')
plt.show()
r2squared=r2_score(T1,f(t,*p))
print('R2 is ' +str(r2squared)+' by curvefitting')


# In[ ]:




