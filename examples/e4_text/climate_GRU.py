import os

data_file = 'D:/all-dataset/jena_climate_2009_2016.csv'
f = open(data_file, encoding='utf-8')
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print(header)
print(len(lines))

import numpy as np

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

import matplotlib.pyplot as plt

temp = float_data[:, 1]
# plt.plot(range(len(temp)), temp)
# plt.show()
#
# plt.plot(range(1440), temp[: 1440])
# plt.show()

# 标准化数据 每个时间序列减去平均值，再除以标准差
mean = float_data[:20000].mean(axis=0)
print(mean)
float_data -= mean
std = float_data[:20000].std(axis=0)
float_data /= std
rows = np.random.randint(0, 1000, 128)
print(rows)


def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                # 重置
                i = min_index + batch_size
            else:
                rows = np.arange(i, min(i + batch_size, max_index))
                i += len(rows)
        # lookback//step 间隔
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data, lookback=lookback, delay=delay, min_index=0, max_index=200000, shuffle=True,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(float_data, lookback=lookback, delay=delay, min_index=200001, max_index=300000, shuffle=True,
                    step=step,
                    batch_size=batch_size)
test_gen = generator(float_data, lookback=lookback, delay=delay, min_index=300001, max_index=400000, shuffle=True,
                     step=step,
                     batch_size=batch_size)

val_steps = (300000 - 200001 - lookback) // batch_size
test_steps = (len(float_data) - 300001 - lookback) // batch_size

# 使用GPU
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))

from keras import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1]), dropout=0.2, recurrent_dropout=0.2))
# model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1]),dropout=0.1,recurrent_dropout=0.5,return_sequences=True))
# model.add(layers.GRU(64,activation='relu',dropout=0.1,recurrent_dropout=0.5))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')

history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40, validation_data=val_gen,
                              validation_steps=val_steps)

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='train loss')
plt.plot(epochs, val_loss, 'b', label='val loss')
plt.title('train & val loss')
plt.legend()
plt.show()
