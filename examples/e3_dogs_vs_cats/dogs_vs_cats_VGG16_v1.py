import os

from keras import layers, models, activations, optimizers, losses

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

from keras.applications import VGG16
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
conv_base.summary()
datagen = ImageDataGenerator(rescale=1. / 255)
batch_size = 20


def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size:(i + 1) * batch_size] = features_batch
        labels[i * batch_size:(i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels


train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)


def DVCModel():
    model = models.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation=activations.relu))
    model.add(layers.Dense(1, activation=activations.sigmoid))
    model.compile(loss=losses.binary_crossentropy, optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])
    return model


model = DVCModel()
# 多GPU
# from keras.utils import multi_gpu_model
# model = multi_gpu_model(model,gpus=2)
history = model.fit(train_features, train_labels, batch_size=20, epochs=30,
                    validation_data=(validation_features, validation_labels))
model.save('dogs_vs_cats_small.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

import matplotlib.pyplot as plt

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('t&v acc')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('t&v loss')
plt.legend()
plt.show()
