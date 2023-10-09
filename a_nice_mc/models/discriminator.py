import tensorflow as tf
from tensorflow import keras
from keras import layers
# class Discriminator(object):
#     def __init__(self):
#         self.name = 'discriminator'

#     def __call__(self, x):
#         raise NotImplementedError(str(type(self)))

class MLPDiscriminator(object):
    def __init__(self, input_shape, dims):
        #tf.compat.v1.disable_eager_execution()
        self.name = 'discriminator'
        # input = keras.Input(input_shape)
        # x = input
        # for dimension in dims:
        #     x = layers.Dense(dimension)(x)
        #     x = layers.LeakyReLU(alpha = 0.2)(x)
        # output = layers.Dense(1)(x)
        # self.nn = keras.Model(input, output)
        #tf.compat.v1.enable_eager_execution()
        self.nn = keras.Sequential()
        self.nn.add(layers.InputLayer(input_shape = input_shape))
        for dimension in dims:
            self.nn.add(layers.Dense(dimension))
            self.nn.add(layers.LeakyReLU(alpha = 0.2))
        self.nn.add(layers.Dense(1))

    def __call__(self, x):
        y = self.nn(x)
        return y
