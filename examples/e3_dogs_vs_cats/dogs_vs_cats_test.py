import os

import numpy as np
from keras import models

# 原始目录
ori_dataset_dir = 'D:\\all-dataset\\dogs-vs-cats\\train'
# 小数据及目录
small_data_dir = 'D:\\all-dataset\\dogs-vs-cats-small'
# 划分数据集:train|validation|test
train_dir = os.path.join(small_data_dir, 'train')
validation_dir = os.path.join(small_data_dir, 'validation')
test_dir = os.path.join(small_data_dir, 'test')
# cats dogs train
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
# cats dogs validation
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
# cats dogs test
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

# 使用GPU
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))

from keras.preprocessing.image import image

# file_path ='D:\\all-dataset\\dogs-vs-cats-small\\test\\cats\\cat.1500.jpg'
file_path = 'D:\\all-dataset\\dogs-vs-cats-small\\test\dogs\\dog.1500.jpg'
img = image.load_img(file_path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

model = models.load_model('dogs_vs_cats_small.h5')
y = model.predict(x)
print(y)
