#!/usr/bin/env python

"""
Weight initializers typically used in deep neural networks/ cnns.
Inspired by:
https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/contrib/layers/python/layers/initializers.py
"""

import math
import tensorflow as tf
from tensorflow.python.ops import random_ops

__author__ = "Udo Dehm"
__copyright__ = "Copyright 2017"
__credits__ = ["Udo Dehm"]
__license__ = ""
__version__ = "0.1"
__maintainer__ = "Udo Dehm"
__email__ = "udo.dehm@mailbox.org"
__status__ = "Development"


__all__ = ['_xavier_msra', 'xavier_msra_initializer',
           'xavier_msra_initializer_conv2d']


def _xavier_msra(n_inputs, n_outputs, shape, method, seed, dtype):
    """
    Helper function that calculates random variables (in given shape). Depending
    on the given method a variant of normalized uniform (Xavier initialization
    methods) or Gaussian (MSRA initialization method) distribution is used.
    Normalization is done by number of incoming (fan-in) and outgoing (fan-out)
    weight connections.
    The two different weight initialization methods are described in the
    following papers:

    "Kaiming" a.k.a. "MSRA" weight initialization: 
        He, Kaiming, et al. "Delving deep into rectifiers: Surpassing 
        human-level performance on imagenet classification." arXiv preprint
        arXiv:1502.01852 (2015).
    This initialization scheme is well suited for layers that are followed
        by a ReLU non-linear unit, source: 
        https://arxiv.org/pdf/1502.01852v1.pdf
    
    Xavier weight initialization: 
        Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of
        training deep feedforward neural networks". International conference on
        artificial intelligence and statistics. 2010.
    Initial weights are randomly picked from a U[-sqrt(3/n),sqrt(3/n)],
        where n is the fan-in (default), fan-out, or their average. According to
        the paper this initialization is particularly well suited for layers
        that are followed by a Sigmoid activation unit. 
    
    Other sources:
        - Andi's blog: http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
        - also tensorflow implementations of functions:
          tf.contrib.layers.xavier_initializer(uniform=True, seed=None,
                                               dtype=tf.float32)
    Args:
        n_inputs: number of incoming node connections/ weights (fan_in)
        n_outputs: number of outgoing node connections/ weights (fan_out)
        shape: specifies the shape of the random variables
        method: flag which specifies the method to use. Valid arguments are:
                'xavier_tf_caffe', 'msra_tf', 'xavier_torch', 'msra_caffe_torch'
        seed: seed for generating random variables
        dtype: type of random variables to generate
    Returns:
        Random variables that represent the weight initializations.
        These variables are sampled according to the used method.
    """

    if method=='xavier_tf_caffe':
        # 6 was used in the paper.
        init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
        return random_ops.random_uniform(shape, -init_range, init_range,
                                         dtype, seed=seed)
    elif method=='msra_tf':
        # 3 gives us approximately the same limits as above since this repicks
        # values greater than 2 standard deviations from the mean.
        stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
        return random_ops.truncated_normal(shape, 0.0, stddev, dtype, seed=seed)
    elif method=='xavier_torch':
        # 6 was used in the paper.
        init_range = math.sqrt(2.0 / (n_inputs + n_outputs))
        return random_ops.random_uniform(shape, -init_range, init_range,
                                         dtype, seed=seed)
    elif method=='msra_caffe_torch':
        # 3 gives us approximately the same limits as above since this repicks
        # values greater than 2 standard deviations from the mean.
        stddev = math.sqrt(4.0 / (n_inputs + n_outputs))
        return random_ops.truncated_normal(shape, 0.0, stddev, dtype, seed=seed)
    else:
        raise ValueError("Method must be one of the following: " + 
                         "'xavier_tf_caffe', 'msra_tf', 'xavier_torch', " +
                         "'msra_caffe_torch'")


def xavier_msra_initializer(method='xavier_tf_caffe', seed=None,
                            dtype=tf.float32):
    """
    Returns an initializer performing "Xavier" or "MSRA" initialization for
    weights.

    For further details see function _xavier_msra().

    This initializer is designed to keep the scale of the gradients roughly the
    same in all layers. In uniform distribution (Xavier) this ends up being the
    range: `x = sqrt(6. / (in + out)); [-x, x]` and for normal distribution
    (MSRA) a standard deviation of `sqrt(3. / (in + out))` is used.

    The returned initializer assumes the shape of the weight matrix to be
    initialized is `[in, out]`.

    Args:
        method: flag which specifies the method to use. Valid arguments are:
                'xavier_tf_caffe', 'msra_tf', 'xavier_torch', 'msra_caffe_torch'
        seed: A Python integer. Used to create random seeds.
        dtype: The data type. Only floating point types are supported.

    Returns:
        An initializer for a 2-D weight matrix.

    Raises:
        TypeError: If dtype is not a floating point type.
    """
    if not dtype.is_floating:
        raise TypeError('Cannot create Xavier initializer for non-floating ' +
                        'point type.')
    def _initializer(shape, dtype=dtype, partition_info=None):
        n_inputs = shape[0]
        n_outputs = shape[1]
        return _xavier_msra(n_inputs, n_outputs, shape, method, seed, dtype)
    return _initializer


def xavier_msra_initializer_conv2d(method='msra_caffe_torch', seed=None,
                                   dtype=tf.float32):
    """
    Returns an "Xavier" or "MSRA" initializer for 2D convolution weights.

    For details on the initialization performed, see `xavier_initializer`. This
    function initializes a convolution weight variable which is assumed to be
    4-D. The first two dimensions are expected to be the kernel size, the third
    dimension is the number of input channels, and the last dimension is the
    number of output channels.

    The number of inputs is therefore `shape[0]*shape[1]*shape[2]`, and the
    number of outputs is `shape[0]*shape[1]*shape[3]`.

    Args:
        method: flag which specifies the method to use. Valid arguments are:
                'xavier_tf_caffe', 'msra_tf', 'xavier_torch', 'msra_caffe_torch'
        seed: A Python integer. Used to create random seeds. 
        dtype: The data type. Only floating point types are supported.
        
    Returns:
        An initializer for a 4-D weight matrix.

    Raises:
        TypeError: If dtype is not a floating point type.
    """
    if not dtype.is_floating:
        raise TypeError('Cannot create Xavier initializer for non-floating ' +
                        'point type.')
    def _initializer(shape, dtype=dtype, partition_info=None):
        # fan in (kernel_height * kernel_width * in_channel):
        n_inputs = shape[0] * shape[1] * shape[2]
        # fan out (kernel_height * kernel_width * out_channel):
        n_outputs = shape[0] * shape[1] * shape[3]
        return _xavier_msra(n_inputs, n_outputs, shape, method, seed, dtype)
    return _initializer


if __name__ == "__main__":
    # example of how to apply the xavier_msra initializer.
    # here: use the 'msra_caffe_torch' method for a convolutional shape:
    weights1 = tf.get_variable('weights_example1',
                               [2, 3, 3, 3],
                               initializer=xavier_msra_initializer_conv2d(\
                                       method='msra_caffe_torch', seed=None,
                                       dtype=tf.float32))
    weights2 = tf.get_variable('weights_example2', [2, 3, 3, 3], 
                               initializer=xavier_msra_initializer_conv2d(\
                                       method='xavier_tf_caffe', seed=None,
                                       dtype=tf.float32))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Weights initialized with method 'msra_caffe_torch' for conv2d" +
              "shapes:")
        print(weights1.eval())
        print("Weights initialized with method 'xavier_tf_caffe' for conv2d" +
              "shapes:")
        print(weights2.eval())

