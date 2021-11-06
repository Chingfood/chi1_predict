#!/usr/bin/env python
# coding: utf-8

# In[99]:


import numpy as np
import os
import pickle
import copy


# In[100]:


PSSM_directory = "/nfs/amino-home/qingyliu/dihedral_angle/PSSM_library"
name_file_path = "/nfs/amino-home/qingyliu/dihedral_angle/template_list"
secondary_structure_directory =  "/nfs/amino-home/qingyliu/dihedral_angle/PSSM_library"
chi1_directory= "/nfs/amino-home/qingyliu/dihedral_angle/chi1_dataset"
chi1_native_distribution = "/home/qingyliu/test/Chi1_native.pkl"


# PSSM_directory = "/home/qingyliu/test"#"/home/chingyuenliu/Documents/test"
# 
# name_directory = "/home/qingyliu/test"#"/home/chingyuenliu/Documents/test"
# secondary_structure_directory = "/home/qingyliu/test"#"/home/chingyuenliu/Documents/test"#
# chi1_directory= "/home/qingyliu/test"#"/home/chingyuenliu/Documents/test"#
# chi1_native_distribution = "/home/qingyliu/test/Chi1_native.pkl"#"/home/chingyuenliu/Documents/test/Chi1_native.pkl"#

# PSSM_directory = "/home/chingyuenliu/Documents/test"
# 
# name_directory = "/home/chingyuenliu/Documents/test"
# secondary_structure_directory = "/home/chingyuenliu/Documents/test"#
# chi1_directory= "/home/chingyuenliu/Documents/test"#
# chi1_native_distribution = "/home/chingyuenliu/Documents/test/Chi1_native.pkl"#

# In[109]:


def aa_convert(aa):
    three_aa = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
    one_aa = {'C': 'CYS', 'D': 'ASP', 'S': 'SER', 'Q': 'GLN', 'K': 'LYS', 'I': 'ILE',
    'P': 'PRO', 'T': 'THR', 'F': 'PHE', 'N': 'ASN',  'G': 'GLY',  'H': 'HIS',  'L': 'LEU',
     'R': 'ARG', 'W': 'TRP', 'A': 'ALA', 'V': 'VAL', 'E': 'GLU', 'Y': 'TYR', 'M': 'MET'}
    aa=aa.upper()
    if len(aa) > 1:
        try:
            return three_aa[aa]
        except:
            if aa == "SEC" :
                return 'C'
            elif aa == "PYL":
                return 'K'
            else:
                return "X"
    else:
        try: 
            return one_aa[aa]
        except:
            return 'UNK'


# In[102]:


def PSSM_writing(dataset,filename,directory_path):
    dataset.append([])
    with open(os.path.join(directory_path,filename.replace('\n','')+".mtx")) as f:
        seq_len = int(f.readline())
        seq = f.readline()
        flag_1 = 2
        for line in f:
            flag_1 += 1
            if line.startswith("-32768"):
                dataset[-1].append(np.zeros(20))
                line = line.replace('\n','').split()
                flag = 0
                for i in range(1,23):
                    if i == 2 or i == 21:
                        continue;
                    else:
                        try:
                            dataset[-1][-1][flag] = float(line[i])
                            flag += 1
                        except:
                            print(filename)
                            print(flag_1)
    return seq_len, seq
                            
                
                


# In[103]:


def secondary_structure_writing(dataset,filename,directory_path):
    with open(os.path.join(directory_path,filename.replace('\n','')+".ss2")) as f:
        flag_2 = 0
        
        for i in range(2):
            f.readline()
        
        for line in f:
            
            dataset[-1][flag_2] = np.hstack((dataset[-1][flag_2],np.zeros(3)))
            line = line.replace('\n','').split()
            for i in range(3):
                dataset[-1][flag_2][20+i] = float(line[i+3])
            flag_2 += 1


# In[104]:


def native_distribution(dataset,seq_len,seq,chi1_native = chi1_native_distribution):
    f1 = open(chi1_native,"rb")
    b = pickle.load(f1)
    f1.close()
    b['GLY'] = [1/3,1/3,1/3]
    b['ALA'] = [1/3,1/3,1/3]
    b['UNK'] = [0.96,0.03,0.01]
    if (seq_len == len(dataset[-1])):
        for i in range(seq_len):
            dataset[-1][i] = np.hstack((dataset[-1][i],np.zeros(3)))
            dataset[-1][i][-3:] = b[aa_convert(seq[i])]
        
    else:
        return False
    
    return True
        


# In[105]:


def terminal_residue(dataset):
    front_dataset = copy.deepcopy(dataset)
    back_dataset = copy.deepcopy(dataset)
    for j in range(len(dataset)):
        for i in range(len(dataset[j])):
            dataset[j][i] = np.insert(dataset[j][i],-1,0)
            front_dataset[j][i] = np.insert(front_dataset[j][i],-1,0)
            back_dataset[j][i] = np.insert(back_dataset[j][i],-1,0)
            if i <7 or i >= len(dataset[j]) - 7:
                dataset[j][i][-2] = 1
            if i < 14:
                front_dataset[j][i][-2] = 1
            if i > len(dataset[j]) - 15 :
                back_dataset[j][i][-2] = 1
    for j in range(len(dataset)):
        for i in range(14):
            front_dataset[j].insert(0,np.zeros(len(front_dataset[0][0])))
            back_dataset[j].append(np.zeros(len(back_dataset[0][0])))
        for k in range(7):
            dataset[j].insert(0,np.zeros(len(dataset[0][0])))
            dataset[j].append(np.zeros(len(dataset[0][0])))
    return front_dataset, back_dataset


# In[106]:


def chi1_value(dataset,filename,directory_path,seq):
    with open(os.path.join(directory_path,filename.replace('\n',''))) as f:
        flag_3 = 0
        for i in range(len(dataset[-1])):
            dataset[-1][i] = np.hstack((dataset[-1][i],np.zeros(1)))
        for line in f:
            line = line.replace('\n','').split()
            if flag_3 < seq_len and aa_convert(aa_convert(seq[flag_3])) == aa_convert(line[0]):
                if line[0] == "ALA" or line[0] == "GLY":
                    dataset[-1][flag_3][-1] = 940501 #the residue is ala or gly
                else:
                    if (len(line) > 1):
                        dataset[-1][flag_3][-1] = float(line[1])
                    else:
                        dataset[-1][flag_3][-1] = 930524 # the residue is missing coordinate to calculate chi1
            else:
                #print("Chi1 error not the same")
                #print(filename)
                #print(flag_3+1)
                continue
            flag_3 += 1


# In[111]:


dataset = []
count =0
with open(name_file_path) as f:
    for line in f:
        seq_len,seq = PSSM_writing(dataset,line.replace('\n',''),PSSM_directory)
        secondary_structure_writing(dataset,line.replace('\n',''),secondary_structure_directory)
        if (not native_distribution(dataset,seq_len,seq)):
            print("native distribution")
            print(line)
        
        chi1_value(dataset,line.replace('\n',''),chi1_directory,seq)  


front_dataset,back_dataset = terminal_residue(dataset)
        
    
        
        


# In[ ]:


f = open("middle_dataset.pkl","wb")
pickle.dump(dataset,f)
f.close()
f1 = open("front_dataset.pkl","wb")
pickle.dump(front_dataset,f)
f1.close()
f2 = open("back_dataset.pkl","wb")
pickle.dump(back_dataset,f)
f2.close()

