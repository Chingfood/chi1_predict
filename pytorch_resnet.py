#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from torch.utils.data import Dataset


# In[2]:


class TrialDataset(Dataset):
    #Characterizes a dataset for PyTorch'
    def __init__(self, feature_path, label_path, name_list, channel="first",transform = False):
        'Initialization'
        self.feature_path = feature_path
        self.label_path = label_path
        self.name_list = name_list
        if (channel != "first"):
            self.channel = "last"
        else:
            self.channel = channel
        if transform == False:
            self.transform = transform
        else:
            self.transform = True

    def __len__(self):
        'Denotes the total number of samples'
        return np.shape(self.name_list)[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        #ID = self.list_IDs[index]

        # Load data and get label
        #X = torch.load('data/' + ID + '.pt')
        
        #y = self.labels[ID]
        feature = np.load(os.path.join(self.feature_path,self.name_list[index]))
        label = np.load(os.path.join(self.label_path,self.name_list[index]))
        if self.transform == True:
            feature = np.squeeze(feature,0)
        if self.channel == "last":
            feature = np.rollaxis(feature,-1,0)

        return feature, label #, name_list[index]


# In[3]:


name_list = "/nfs/amino-home/qingyliu/dihedral_angle/input_name.npy"
feature_path = "/nfs/amino-home/qingyliu/dihedral_angle/conv_ML_input"
label_path = "/nfs/amino-home/qingyliu/dihedral_angle/temp"
name_list = np.load(name_list)


# In[4]:


training_set = TrialDataset(feature_path,label_path,name_list,"last",True)


# In[5]:


import torch
import torch.nn as nn
# CUDA for PyTorch
#use_cuda = torch.cuda.is_available()
#device = torch.device("cuda:0" if use_cuda else "cpu")
#cudnn.benchmark = True


# In[6]:


# Parameters
params = {'batch_size': 1,
          'shuffle': False,
          'num_workers': 1}
max_epochs = 100


# In[7]:


training_generator = torch.utils.data.DataLoader(training_set, **params)


# In[8]:


def padding_size(kernel_size,dilation=1,stride=1,input_size = 1):
    pad = ((input_size -1)*(stride-1) + dilation*(kernel_size -1))/2
    if pad.is_integer():
        return int(pad)
    else:
        raise NameError('value is not integer')


# In[9]:


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, input_size = 1, padding = "same"):
    """3x3 convolution with padding"""
    if padding == "valid":
        pad = 0
    else:
        pad = padding_size(3,dilation,stride,input_size)
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, 
                     padding= pad, bias=False, dilation=dilation)


# In[10]:


class IdentityPadding(nn.Module):
    def __init__(self, num_filters, channels_in, stride):
        super(IdentityPadding, self).__init__()
        # with kernel_size=1, max pooling is equivalent to identity mapping with stride
        self.identity = nn.MaxPool2d(1, stride=stride)
        self.num_zeros = num_filters - channels_in
    
    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.num_zeros))
        out = self.identity(out)
        return out

# option B from paper
class ConvProjection(nn.Module):

    def __init__(self, num_filters, channels_in, stride):
        super(ResA, self).__init__()
        self.conv = nn.Conv2d(channels_in, num_filters, kernel_size=1, stride=stride)
    
    def forward(self, x):
        out = self.conv(x)
        return out

# experimental option C
class AvgPoolPadding(nn.Module):

    def __init__(self, num_filters, channels_in, stride):
        super(AvgPoolPadding, self).__init__()
        self.identity = nn.AvgPool2d(stride, stride=stride)
        self.num_zeros = num_filters - channels_in
    
    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.num_zeros))
        out = self.identity(out)
        return out


# In[11]:


class ResBlock(nn.Module):
    
    def __init__(self, num_filters, channels_in=None, dilation = 1, stride=1, res_option='B', use_dropout=False):
        super(ResBlock, self).__init__()
        
        # uses 1x1 convolutions for downsampling
        if not channels_in or channels_in == num_filters:
            channels_in = num_filters
            self.projection = None
        else:
            if res_option == 'A':
                self.projection = IdentityPadding(num_filters, channels_in, stride)
            elif res_option == 'B':
                self.projection = ConvProjection(num_filters, channels_in, stride)
            elif res_option == 'C':
                self.projection = AvgPoolPadding(num_filters, channels_in, stride)
        self.use_dropout = use_dropout

        self.conv1 = conv3x3(channels_in, num_filters, dilation=dilation, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels_in, num_filters, dilation=dilation, stride=stride)
        self.bn2 = nn.BatchNorm2d(num_filters)
        if self.use_dropout:
            self.dropout = nn.Dropout(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_dropout:
            out = self.dropout(out)
        if self.projection:
            residual = self.projection(x)
        out += residual
        out = self.relu2(out)
        return out


# In[12]:


class ResNet(nn.Module):
    
    def __init__(self, n=61, res_option='B', use_dropout=False):
        super(ResNet, self).__init__()
        self.res_option = res_option
        self.use_dropout = use_dropout
        self.conv1 = nn.Conv2d(526, 64, kernel_size=1, stride=1)
        self.norm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.layers1 = self._make_layer(n, 64, 64, 1)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=1, stride=1)
        self.norm2 = nn.BatchNorm2d(3)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv1d(3,3,kernel_size =1)
        self.softmax = nn.Softmax(1)
    def _make_layer(self, layer_count, channels, channels_in, stride):
        return nn.Sequential(
            ResBlock(channels, channels_in, stride, res_option=self.res_option, use_dropout=self.use_dropout),
            *[ResBlock(channels,channels_in, 2**(n%5) ) for n in range(1,layer_count)])
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        
        out = self.layers1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        
        out = (torch.mean(out,2) + torch.mean(out,3))*0.5
        
        out = self.conv3(out)
        out = self.softmax(out)

        return out


# In[13]:


def customLoss(yPred,yTrue):

    yPred= torch.clamp(yPred, 1e-6, (1. - 1e-6))
    mask= (yTrue <= 2).squeeze_(0)
    
    return torch.nn.CrossEntropyLoss()(yPred[:,:,mask],yTrue[:,mask])


# In[14]:


# defining the model
model = ResNet(n=6)
# defining the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.07)
# defining the loss function
#criterion = torch.nn.CrossEntropyLoss()
# checking if GPU is available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# if torch.cuda.is_available():
#     model = model.cuda()
#     criterion = criterion.cuda()
if use_cuda:
    model.to(device)
    #criterion.to(device)
print(model)


# In[15]:


def train(epoch):
    model.train()    # Training
    for idx, (local_batch, local_labels) in enumerate(training_generator):
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        optimizer.zero_grad()
        output_train = model(local_batch.float())
        loss_train = customLoss(output_train, local_labels)
        loss_train.backward()
        optimizer.step()
        tr_loss = loss_train.item()
        if (index%4000 == 0):
            print('Epoch : ',epoch+1, '\t', 'loss :', tr_loss)


# In[16]:


max_epochs = 2
# Loop over epochs
for epoch in range(max_epochs):
    train(epoch)


# 
# # Validation
# with torch.set_grad_enabled(False):
#     for local_batch, local_labels in validation_generator:
#         # Transfer to GPU
#         local_batch, local_labels = local_batch.to(device), local_labels.to(device)
