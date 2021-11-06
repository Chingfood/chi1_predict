#!/usr/bin/env python
# coding: utf-8

# In[13]:


folder_path = "/nfs/amino-home/qingyliu/dihedral_angle/temp"


# In[14]:


import os 
list_name = []
directory = os.fsencode(folder_path)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    list_name.append(filename)


# In[15]:


import numpy as np
list_npy = np.asarray(list_name)


# In[9]:


type(list_npy[0])


# In[10]:


a = np.load(os.path.join(folder_path,list_npy[0]))


# In[11]:


a.shape


# In[16]:


np.save("/nfs/amino-home/qingyliu/dihedral_angle/input_name.npy",list_npy)

