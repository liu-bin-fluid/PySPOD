#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf


# In[ ]:

###from tensorflow.python.keras.layers import layers, Dense
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential


# In[1]:


import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)


# In[ ]:


image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)


# In[ ]:


roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]))


# In[ ]:


PIL.Image.open(str(roses[1]))


# In[ ]:


tulips = list(data_dir.glob('tulips/*'))
PIL.Image.open(str(tulips[0]))


# In[ ]:


PIL.Image.open(str(tulips[1]))


# In[ ]:


#creation of dataset

batch_size = 32
img_height = 180
img_width = 180


# In[ ]:


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.999,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# In[ ]:


tf.version


# In[ ]:




