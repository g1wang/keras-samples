import numpy as np
from keras.datasets import imdb


def decode(data, word_index):
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decode_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in data])
    return decode_review


def vectorize_sequence(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


from keras import models
from keras import layers
from keras import optimizers, losses, metrics, regularizers


def train():
    (train_data, train_lables), (test_data, test_lables) = imdb.load_data(num_words=10000)
    word_index = imdb.get_word_index()
    x_train = vectorize_sequence(train_data)
    x_test = vectorize_sequence(test_data)

    y_train = np.asarray(train_lables).astype('float32')
    y_test = np.asarray(test_lables).astype('float32')

    model = models.Sequential()
    model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu', input_shape=(10000,)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    # model.add(layers.Dense(1, activation='softmax'))

    model.compile(optimizer=optimizers.RMSprop(lr=0.01), loss=losses.binary_crossentropy,
                  metrics=[metrics.binary_accuracy])

    model.fit(x_train, y_train, epochs=20, batch_size=256, validation_data=(x_test, y_test))
    model.save('imdb.pkl')


def test():
    (train_data, train_lables), (test_data, test_lables) = imdb.load_data(num_words=10000)
    word_index = imdb.get_word_index()
    x_test = vectorize_sequence(test_data)
    y_test = np.asarray(test_lables).astype('float32')
    model = models.load_model('imdb.pkl')
    decode_review = decode(test_data[1])
    print(decode_review)
    testDatas = np.zeros((1, 10000))
    testDatas[0] = x_test[1]
    result = model.predict(testDatas)
    print(result);


if __name__ == '__main__':
    train()
    # test()
