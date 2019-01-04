from keras import layers, models, activations, optimizers, losses, metrics


def Model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation=activations.relu, input_shape=(28, 28, 1)))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation=activations.relu))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation=activations.relu))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation=activations.relu))
    model.add(layers.Dense(10, activation=activations.softmax))
    model.compile(optimizer=optimizers.RMSprop(), loss=losses.categorical_crossentropy,
                  metrics=[metrics.categorical_accuracy])

    return model


from keras.datasets import mnist
from keras.utils import to_categorical


def train():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255.
    train_labels = to_categorical(train_labels)

    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255.
    test_labels = to_categorical(test_labels)

    model = Model()
    model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))
    print(model.evaluate(test_images, test_labels))
    model.save('cnn-mnist.pkl')


def test():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255.
    test_labels = to_categorical(test_labels)
    model = models.load_model('cnn-mnist.pkl')
    model.summary()
    print(model.evaluate(test_images, test_labels))


if __name__ == '__main__':
    train()
    # test()
