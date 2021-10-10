import tensorflow as tf
#import tensorflow.contrib.layers as tcl
import tf_slim as slim

def dense(inputs, num_outputs, activation_fn=tf.identity, normalizer_fn=None, normalizer_params=None):
    return slim.fully_connected(inputs, num_outputs, activation_fn=activation_fn,
                               normalizer_fn=normalizer_fn, normalizer_params=normalizer_params)


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


lrelu = leaky_relu
