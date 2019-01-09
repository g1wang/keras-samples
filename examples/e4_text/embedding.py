from keras.datasets import imdb
from keras import preprocessing, models, layers
import numpy as np
import time
time_stape = round(time.time()*1000)
max_features = 10000
maxlen = 100

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

model = models.Sequential()
model.add(layers.Embedding(max_features, 8, input_length=maxlen))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
models.save_model(model,'embedding_imdb.h5')
trained_model = models.load_model('embedding_imdb.h5')
result = trained_model.predict(x_test)
print(trained_model.evaluate(x_test,y_test))
print(result[3])
print(y_test[3])
