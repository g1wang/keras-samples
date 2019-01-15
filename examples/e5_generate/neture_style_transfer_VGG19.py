# 使用GPU
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))

# 定义初始变量
from keras.preprocessing.image import load_img, img_to_array

target_image_path = 'img/protrait.jpg'
style_reference_image_path = 'img/style_reference_image_path.jpg'

width, height = load_img(target_image_path).size
img_height = 400
img_width = int(width * img_height / height)

import numpy as np
from keras.applications import vgg19


# 对进出VGG19卷积网络的图像加载、预处理和后处理
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


# vgg19.preprocess_input 逆操作
def deprocess_image(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# 加载预训练的VGG19

from keras import backend as K

target_image = K.constant(preprocess_image(target_image_path))
style_reference_image = K.constant(preprocess_image(style_reference_image_path))
combination_image = K.placeholder((1, img_height, img_width, 3))
print(target_image.shape)
input_tensor = K.concatenate([target_image, style_reference_image, combination_image], axis=0)
print(input_tensor.shape)
model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)

model.summary()


# 内容损失，要保证目标图像和生成图像在VGG顶层具有相似的结果

def content_loss(base, combination):
    return K.sum(K.square(combination - base))


# 风格损失

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def total_variation_loss(x):
    a = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :])
    b = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, img_height - 1:, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


# 需最小化的最终损失
output_dict = dict([(layer.name, layer.output) for layer in model.layers])
content_layer = 'block5_conv2'
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
# 损失分量的加权比重
total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.05

loss = K.variable(0.)
layer_features = output_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_image_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(target_image_features, combination_image_features)

for layer_name in style_layers:
    layer_features = output_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(style_layers)) * sl
loss += total_variation_weight * total_variation_loss(combination_image)

# 梯度计算

grads = K.gradients(loss, combination_image)[0]
fetch_loss_and_grads = K.function([combination_image], [loss, grads])


class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grads_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grads_values)
        self.loss_value = None
        self.grads_values = None
        return grad_values


evaluator = Evaluator()

# 风格转移循环

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time

result_prefix = 'myresult'
iterations = 20
x = preprocess_image(target_image_path)
x = x.flatten()

for i in range(iterations):
    print('start ', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxfun=20)
    print('current loss value:', min_val)
    img = x.copy().reshape((img_height, img_width, 3))
    fname = 'neture_style_transfer_VGG19_img/' + result_prefix + '%d.png' % i
    imsave(fname, img)
    print('saved', i)
    end_time = time.time()
    print('iterations %d completed in %ds ' % (i, end_time - start_time))
