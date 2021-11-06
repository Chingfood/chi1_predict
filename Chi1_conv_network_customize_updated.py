from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from tensorflow import keras

import numpy as np

import tensorflow.keras.backend as K


def customLoss(yTrue,yPred):

    if yTrue.shape[0] == None:
        return 1e-7

    yPred= tf.clip_by_value(yPred, 1e-7, (1. - 1e-7))
    mask=K.less_equal(yTrue,2)
    
    return tf.reduce_mean(K.categorical_crossentropy(tf.one_hot(tf.cast(tf.boolean_mask(yTrue, mask),tf.int32), 3),tf.boolean_mask(yPred, mask)))





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
    model.add(keras.layers.Conv1D(3, kernel_size=5,padding = "Same", activation=tf.keras.activations.softmax)) 

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=customLoss,
                )

    return model





def input_generator(input_feature, input_label,epochs):

    for j in range(epochs):
        for i in range(input_label.shape[0]):
            shape = list(input_feature[i].shape)
            shape.insert(0,-1)
            yield input_feature[i].reshape(shape), input_label[i].reshape(-1,input_label[i].shape[0])
            
            
            
            
train_feature_file_format = "/oasis/projects/nsf/mia174/qingyliu/ML_data/whole_seq/hhblits/train_feature_removed.npy"

train_label_file_format = "/oasis/projects/nsf/mia174/qingyliu/ML_data/whole_seq/hhblits/train_label_removed.npy"

train_features= np.load(train_feature_file_format,allow_pickle=True)
train_labels = np.load(train_label_file_format,allow_pickle=True)



checkpoint_path_all_conv = "/oasis/projects/nsf/mia174/qingyliu/ML_data/whole_seq/hhblits/training_checkpoint/cp.ckpt"
checkpoint_dir_all_conv = os.path.dirname(checkpoint_path_all_conv)
cp_callback_all_conv = tf.keras.callbacks.ModelCheckpoint(checkpoint_path_all_conv,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq=4500        )

model_all_conv = create_model_all_conv()


# In[59]:



#model_all_conv.load_weights(checkpoint_path_all_conv)


# In[ ]:



model_all_conv.fit_generator(input_generator(train_features, train_labels,epochs = 20),
                             steps_per_epoch = train_labels.shape[0],
                             epochs = 20,
                             callbacks = [cp_callback_all_conv])



    