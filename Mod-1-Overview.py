#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install gekko')


# In[3]:


# install tclab
try:
    import tclab
except:
    # Needed to communicate through usb port
    get_ipython().system('python -m pip install --user pyserial')
    # The --user is put in for accounts without admin privileges
    get_ipython().system('python -m pip install --user tclab ')
    # restart kernel if this doesn't import
    try:
        import tclab
    except:
        print('Restart kernel from menu if Dead kernel')
        print('Restart kernel automatically...')
        import IPython
        app = IPython.Application.instance()
        app.kernel.do_shutdown(restart=True) 


# In[3]:


#Warm-up exercise; turning on LED for 5 seconds
import tclab
import time

with tclab.TCLab() as lab:
    lab.LED(100)
    time.sleep(5.0) #wait 5.0 seconds
    lab.LED(0)


# In[5]:


#Turn on heater Q1. Reads temperature T1 and then again after 10 seconds
with tclab.TCLab() as lab:
    print(lab.T1) # print temperature 1
    lab.Q1(100) #turn on Q1 to 100%
    time.sleep(15) #sleep for 15 seconds
    print(lab.T1) #print temperature 1


# In[11]:


#Blink LED 5 times for 1 seconds each
with tclab.TCLab() as lab:
    for i in range(5): #repeat 5 times
        lab.LED(100)
        time.sleep(1)
        lab.LED(0)
        time.sleep(0.5)


# In[14]:


#Turn on heater to 80% until temperature reaches 50 oC. Update the LED blink time (t). t =(50-T1)/10
with tclab.TCLab() as lab:
    lab.Q1(80)
    while lab.T1<50:
        lab.LED(100)
        time.sleep((50-lab.T1)/10)
        lab.LED(0)
        time.sleep(0.2)


# In[ ]:





# In[ ]:




