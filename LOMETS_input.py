#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[ ]:


fasta_library = "/nfs/amino-home/qingyliu/dihedral_angle/fasta_library_30"
lomet_directory = "/nfs/amino-home/qingyliu/dihedral_angle/LOMETS_input"


# fasta_library ="/home/chingyuenliu/fasta_library"
# lomet_directory = "/home/chingyuenliu/LOMETS_library"

# In[4]:


directory = os.fsencode(fasta_library)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".fasta"):
        os.mkdir(os.path.join(lomet_directory, filename[:-6]))
        os.popen("cp {} {}".format(os.path.join(fasta_library,filename),os.path.join(lomet_directory,filename[:-6],"seq.fasta")))
    

