#!/usr/bin/env python
# coding: utf-8

# In[86]:


import numpy as np


# In[87]:


from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from tensorflow import keras

tf.__version__


# import tensorflow.keras.backend as K
# def customLoss(yTrue,yPred):
#     print("yTrue shape\n")
#     print(yTrue.shape)
#     print("yTrue value\n")
#     print(yTrue)
#     print("yPred shape\n")
#     print(yPred.shape)
#     print("yPred value\n")
#     print(yPred)
#     a = 0
#     if type(yTrue.shape[0]) != int:
#         return 0
#     for i in range(yTrue.shape[0]):
#         a += -(np.log(yPred[i][yTrue[i]]))
#     a = a/yTrue.shape[0]
#     return a

# In[104]:


import tensorflow.keras.backend as K
def customLoss(yTrue,yPred):
    if yTrue.shape[0] == None:
        return 1e-7
    temp = []
    yPred_unpack = tf.unstack(yPred,axis = 1)

    for num,a in enumerate(yPred_unpack):
        if yTrue[0,num] >2:
            continue
        b = K.softmax(a)
        temp.append(K.categorical_crossentropy(yTrue[0,num],b))


    temp_tensor = tf.concat(temp,0)

    return tf.reduce_mean(temp_tensor)
  


# In[101]:


def create_model_all_conv():
 
    model = tf.keras.models.Sequential()
#convolutional layer with rectified linear unit activation
    model.add(keras.layers.Conv1D(60, kernel_size=1,padding = "Same",
                 activation=tf.keras.activations.relu,
                 input_shape=(None,26)))
#32 convolution filters used each of size 5
#again
    model.add(keras.layers.Conv1D(65, kernel_size=5, padding = "Same", activation=tf.keras.activations.relu))
    model.add(keras.layers.Conv1D(70, kernel_size=5, padding = "Same", activation=tf.keras.activations.relu))
    model.add(keras.layers.Conv1D(75, kernel_size=5, padding = "Same", activation=tf.keras.activations.relu))
    model.add(keras.layers.Conv1D(80, kernel_size=5, padding = "Same", activation=tf.keras.activations.relu))
    model.add(keras.layers.Conv1D(120, kernel_size=5, padding = "Same", activation=tf.keras.activations.relu))
    model.add(keras.layers.Conv1D(240, kernel_size=5, padding = "Same", activation=tf.keras.activations.relu))
    model.add(keras.layers.Conv1D(480, kernel_size=5, padding = "Same", activation=tf.keras.activations.relu))
    model.add(keras.layers.Conv1D(240, kernel_size=5, padding = "Same", activation=tf.keras.activations.relu))
    model.add(keras.layers.Conv1D(120, kernel_size=5, padding = "Same", activation=tf.keras.activations.relu))
    model.add(keras.layers.Conv1D(60, kernel_size=5, padding = "Same", activation=tf.keras.activations.relu))
    model.add(keras.layers.Conv1D(30, kernel_size=5,padding = "Same", activation=tf.keras.activations.relu))    
    model.add(keras.layers.Conv1D(25, kernel_size=5,padding = "Same", activation=tf.keras.activations.relu)) 
    model.add(keras.layers.Conv1D(21, kernel_size=5,padding = "Same", activation=tf.keras.activations.relu)) 
    model.add(keras.layers.Conv1D(18, kernel_size=5,padding = "Same", activation=tf.keras.activations.relu)) 
    model.add(keras.layers.Conv1D(15, kernel_size=5,padding = "Same", activation=tf.keras.activations.relu)) 
    model.add(keras.layers.Conv1D(12, kernel_size=5,padding = "Same", activation=tf.keras.activations.relu)) 
    model.add(keras.layers.Conv1D(9, kernel_size=5,padding = "Same", activation=tf.keras.activations.relu))
    model.add(keras.layers.Conv1D(6, kernel_size=5,padding = "Same", activation=tf.keras.activations.relu)) 
    model.add(keras.layers.Conv1D(3, kernel_size=5,padding = "Same")) 

    model.compile(optimizer=tf.keras.optimizers.Adadelta(),
                loss=customLoss,
                )

    return model


# In[102]:


def input_generator(input_feature, input_label):

    for i in range(input_label.shape[0]):
        shape = list(input_feature[i].shape)
        shape.insert(0,-1)
        yield input_feature[i].reshape(shape), input_label[i].reshape(-1,input_label[i].shape[0])


# In[91]:


train_feature_file_format = "/nfs/amino-home/qingyliu/dihedral_angle/ML_data/whole_seq/hhblits/train_feature.npy"

train_label_file_format = "/nfs/amino-home/qingyliu/dihedral_angle/ML_data/whole_seq/hhblits/train_label.npy"

train_features= np.load(train_feature_file_format,allow_pickle=True)
train_labels = np.load(train_label_file_format,allow_pickle=True)


# In[98]:


train_features[0][0]


# In[105]:


checkpoint_path_all_conv = "/home/qingyliu/test/training_checkpoint/cp.ckpt"
checkpoint_dir_all_conv = os.path.dirname(checkpoint_path_all_conv)
cp_callback_all_conv = tf.keras.callbacks.ModelCheckpoint(checkpoint_path_all_conv,
                                                 save_weights_only=True,
                                                 verbose=1)
model_all_conv = create_model_all_conv()


model_all_conv.fit_generator(input_generator(train_features, train_labels),
                             steps_per_epoch = train_labels.shape[0],
                             epochs = 500,
                             callbacks = [cp_callback_all_conv])


# In[ ]:




