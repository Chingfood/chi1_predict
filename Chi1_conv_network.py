#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from tensorflow import keras

tf.__version__


# In[21]:


def create_model_all_conv():
 
    model = tf.keras.models.Sequential()
#convolutional layer with rectified linear unit activation
    model.add(keras.layers.Conv1D(128, kernel_size=1,
                 activation=tf.keras.activations.relu,
                 input_shape=(15,27)))
#32 convolution filters used each of size 5
#again
    model.add(keras.layers.Conv1D(64, kernel_size=3, activation=tf.keras.activations.relu))


    model.add(keras.layers.Conv1D(64, kernel_size=3,strides = 2, activation=tf.keras.activations.relu))    
    model.add(keras.layers.Conv1D(64, kernel_size=4,activation=tf.keras.activations.relu))    
#64 convolution filters used each of size 3
#choose the best features via pooling


 

    model.add(keras.layers.Flatten())
#fully connected to get all relevant data
    model.add(keras.layers.Dense(128, activation=tf.keras.activations.relu))

    model.add(keras.layers.Dense(64, activation=tf.keras.activations.relu))
    model.add(keras.layers.Dense(32, activation=tf.keras.activations.relu))
#output a softmax to squash the matrix into output probabilities
    model.add(keras.layers.Dense(3, activation=tf.keras.activations.softmax))


    model.compile(optimizer=tf.keras.optimizers.Adadelta(),
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])

    return model


# In[5]:


train_feature_file_format = "/nfs/amino-home/qingyliu/dihedral_angle/ML_data/back_tag/back_train_feature_new_3.npy"

train_label_file_format = "/nfs/amino-home/qingyliu/dihedral_angle/ML_data/back_tag/back_train_label_new_3.npy"

train_features= np.load(train_feature_file_format)
train_labels = np.load(train_label_file_format)


# In[7]:


train_labels.shape


# In[22]:


checkpoint_path_all_conv = "/nfs/amino-home/qingyliu/dihedral_angle/ML_data/back_tag/training_all_conv/cp.ckpt"
checkpoint_dir_all_conv = os.path.dirname(checkpoint_path_all_conv)
cp_callback_all_conv = tf.keras.callbacks.ModelCheckpoint(checkpoint_path_all_conv,
                                                 save_weights_only=True,
                                                 verbose=1)
model_all_conv = create_model_all_conv()


model_all_conv.fit(train_features, train_labels,  epochs = 500, batch_size=100,
                callbacks = [cp_callback_all_conv])


# In[25]:


model_all_conv = create_model_all_conv()
model_all_conv.load_weights(checkpoint_path_all_conv)
model_all_conv.fit(train_features, train_labels,  epochs = 500, batch_size=100,
                callbacks = [cp_callback_all_conv])

