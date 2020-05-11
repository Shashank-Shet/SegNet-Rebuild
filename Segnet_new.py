#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, BatchNormalization, Reshape, Lambda, Input, Activation
from tensorflow.keras.models import Model


# In[2]:


def MaxPool2DWithArgmax(input_tensor, ksize, strides):
    p, m = tf.nn.max_pool_with_argmax(input_tensor, ksize=ksize, strides=strides, padding="SAME", include_batch_in_index=True)
    m = K.cast(m, dtype=tf.int32)
    return [p, m]


# In[3]:


def Unpool2D(input_tensors, factor):
    pool, mask = input_tensors
    indices = tf.reshape(mask, (-1,mask.shape[1]*mask.shape[2]*mask.shape[3],1))
    values = tf.reshape(pool, (-1,pool.shape[1]*pool.shape[2]*mask.shape[3]))
    size = tf.size(indices) * factor[1] * factor[2]
    size = tf.reshape(size, [-1])
    t = tf.scatter_nd(indices, values, size)
    t = tf.reshape(t, (-1, mask.shape[1]*factor[1], mask.shape[2]*factor[2], mask.shape[3]))
    return t


# # Encoder Network

# In[7]:


input_shape = (256, 256, 3)
kernel = 3                                                # For the conv operation
args = {"ksize": (1,2,2,1), "strides":(1,2,2,1)}          # For the Maxpool Operation

inputs = Input(input_shape)
x = Conv2D(64, (kernel, kernel), padding="same")(inputs)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Conv2D(64, (kernel, kernel), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

pool_1, mask_1 = Lambda(MaxPool2DWithArgmax, arguments=args)(x)

x = Conv2D(128, (kernel, kernel), padding="same")(pool_1)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Conv2D(128, (kernel, kernel), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

pool_2, mask_2 = Lambda(MaxPool2DWithArgmax, arguments=args)(x)

x = Conv2D(256, (kernel, kernel), padding="same")(pool_2)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Conv2D(256, (kernel, kernel), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Conv2D(256, (1, 1), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

pool_3, mask_3 = Lambda(MaxPool2DWithArgmax, arguments=args)(x)

x = Conv2D(512, (kernel, kernel), padding="same")(pool_3)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Conv2D(512, (kernel, kernel), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Conv2D(512, (1, 1), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

pool_4, mask_4 = Lambda(MaxPool2DWithArgmax, arguments=args)(x)

x = Conv2D(512, (kernel, kernel), padding="same")(pool_4)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Conv2D(512, (kernel, kernel), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Conv2D(512, (1, 1), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

pool_5, mask_5 = Lambda(MaxPool2DWithArgmax, arguments=args)(x)


# # Decoder Network

# In[8]:


pool_size = (1,2,2,1)
n_labels = 10

x = Lambda(Unpool2D, arguments={"factor": pool_size})([pool_5, mask_5])

x = Conv2D(512, (1, 1), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Conv2D(512, (kernel, kernel), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Conv2D(512, (kernel, kernel), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = Lambda(Unpool2D, arguments={"factor": pool_size})([x, mask_4])

x = Conv2D(512, (1, 1), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Conv2D(512, (kernel, kernel), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Conv2D(256, (kernel, kernel), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = Lambda(Unpool2D, arguments={"factor": pool_size})([x, mask_3])

x = Conv2D(256, (1, 1), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Conv2D(256, (kernel, kernel), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Conv2D(128, (kernel, kernel), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = Lambda(Unpool2D, arguments={"factor": pool_size})([x, mask_2])

x = Conv2D(128, (kernel, kernel), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Conv2D(64, (kernel, kernel), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = Lambda(Unpool2D, arguments={"factor": pool_size})([x, mask_1])

x = Conv2D(64, (kernel, kernel), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = Conv2D(n_labels, (1, 1), padding="valid")(x)
x = BatchNormalization()(x)

x = Reshape(
    (input_shape[0] * input_shape[1], n_labels),
    input_shape=(input_shape[0], input_shape[1], n_labels),
)(x)

outputs = Activation("softmax")(x)

model = Model(inputs=inputs, outputs=outputs, name="SegNet")


# In[9]:


model


# In[10]:


model.summary()

