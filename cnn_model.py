 #!/usr/bin/env python

"""
Module that contains the CNN model!
"""

import sys
sys.path.append('/Users/udodehm/workspace/kaggle/util')

import tensorflow as tf
from math import sqrt

import cnn_helpers as cnnhelp
import nn_helpers as nnhelp
import initializers_custom as ic


__author__ = "Udo Dehm"
__copyright__ = "Copyright 2017"
__credits__ = ["Udo Dehm"]
__license__ = ""
__version__ = "0.1"
__maintainer__ = "Udo Dehm"
__email__ = "udo.dehm@mailbox.org"
__status__ = "Development"


__all__ = ['model']   


def model(x,
          training=tf.placeholder(tf.bool),
          dropout_rate=0.5,
          device='/cpu:0'):
    """
    :param x: input to the network
    :type x: tf.placeholder with corresponding shape
    :param training: indicates if network is trained (training -> True) or
        tested/validated (training -> False)
    :param training: np bool or tf bool
    :param dropout_rate: probability that a neuron's output is kept during
        dropout (only during training):
    :type dropout_rate: float (\elem (0, 1])
    :param device: device where to save variables: e.g. standard cpu: '/cpu:0',
        standard gpu: '/gpu:0' (default: '/cpu:0')
    :type device: str
    :return: last layer of network (prediction)
    """
    # create the network:

    # weight initialization methods:
    # 1. use a default normal distribution:
    # weights_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
    # 2. better: use a normalized (normalized by input and output weight
    # connections) initializer which is based on a uniform (Xavier weight
    # initialization method) or a normal (MSRA weight initialization method)
    # distribution.
    # The MSRA initialization scheme is well suited for layers that are followed
    # by a ReLU non-linear unit, while Xavier initializers are particularly well
    # suited for layers that are followed by a Sigmoid activation unit.
    # So it is recommended to use
    weights_initializer = tf.contrib.layers.xavier_initializer(uniform=False,
                                                               seed=None,
                                                               dtype=tf.float32)
    # which is the same as:
    # weights_initializer = ic.xavier_msra_initializer_conv2d(method='xavier_tf_caffe',
    #                                                         seed=None,
    #                                                         dtype=tf.float32)
    # a small variant with different factors (weights have a smaller standard
    # deviation/ witdh) is given by:
    # weights_initializer = ic.xavier_msra_initializer_conv2d(method='msra_caffe_torch',
    #                                                         seed=None,
    #                                                         dtype=tf.float32)
    
    f, s, p, k_in, k_out, name = [12, 4, 'SAME', 3, 96, 'conv1']
    conv_s11 = cnnhelp.conv2d_layer(x,
                                    kernel_shape=[f, f, k_in, k_out],
                                    stride=s,
                                    padding=p,
                                    name_conv_layer=name,
                                    weights_initializer=weights_initializer,
                                    use_bn=True,
                                    training=training,
                                    relu=True,
                                    device=device)

    f, s, p, name = [2, 2, 'SAME', 'max_pool1']
    pool_s11 = tf.nn.max_pool(conv_s11, 
                              ksize=[1, f, f, 1],
                              strides=[1, s, s, 1],
                              padding=p,
                              name=name)


    f, s, p, k_in, k_out, name = [5, 1, 'SAME', k_out, 256, 'conv2']
    conv_s12 = cnnhelp.conv2d_layer(pool_s11,
                                    kernel_shape=[f, f, k_in, k_out],
                                    stride=s,
                                    padding=p,
                                    name_conv_layer=name,
                                    weights_initializer=weights_initializer,
                                    use_bn=True,
                                    training=training,
                                    relu=True,
                                    device=device)

    f, s, p, name = [2, 2, 'SAME', 'max_pool2']
    pool_s12 = tf.nn.max_pool(conv_s12, 
                              ksize=[1, f, f, 1],
                              strides=[1, s, s, 1],
                              padding=p,
                              name=name)
    

    f, s, p, k_in, k_out, name = [3, 1, 'SAME', k_out, 256, 'conv3']
    conv_s13 = cnnhelp.conv2d_layer(pool_s12,
                                    kernel_shape=[f, f, k_in, k_out],
                                    stride=s,
                                    padding=p,
                                    name_conv_layer=name,
                                    weights_initializer=weights_initializer,
                                    use_bn=True,
                                    training=training,
                                    relu=True,
                                    device=device)
    

    f, s, p, k_in, k_out, name = [3, 1, 'SAME', k_out, 384, 'conv4']
    conv_s14 = cnnhelp.conv2d_layer(conv_s13,
                                    kernel_shape=[f, f, k_in, k_out],
                                    stride=s,
                                    padding=p,
                                    name_conv_layer=name,
                                    weights_initializer=weights_initializer,
                                    use_bn=True,
                                    training=training,
                                    relu=True,
                                    device=device)


    f, s, p, k_in, k_out, name = [3, 1, 'SAME', k_out, 256, 'conv5']
    conv_s15 = cnnhelp.conv2d_layer(conv_s14,
                                    kernel_shape=[f, f, k_in, k_out],
                                    stride=s,
                                    padding=p,
                                    name_conv_layer=name,
                                    weights_initializer=weights_initializer,
                                    use_bn=True,
                                    training=training,
                                    relu=True,
                                    device=device)

    f, s, p, name = [2, 2, 'SAME', 'max_pool5']
    pool_s15 = tf.nn.max_pool(conv_s15, 
                              ksize=[1, f, f, 1],
                              strides=[1, s, s, 1],
                              padding=p,
                              name=name)


    f, s, p, k_in, k_out, name = [16, 8, 'SAME', k_out, 256, 'deconv']
    # filter: A 4-D Tensor with the same type as value and shape [height, width,
    # output_channels, in_channels]. filter's in_channels dimension must match
    # that of value.
    #deconv_s1 = conv2d_transpose(value=pool_s15,
    #                          filter=[f, f, output_channels,in_channels],
    #                          output_shape,
    #                          strides,
    #                          padding='SAME',
    #                          data_format='NHWC',
    #                          name=name)
    deconv_s1 = tf.layers.conv2d_transpose(inputs=pool_s15,
                                           filters=k_out,
                                           kernel_size=f,
                                           strides=s,
                                           padding=p,
                                           data_format='channels_last',
                                           activation=None,
                                           use_bias=True,
                                           kernel_initializer=None,
                                           bias_initializer=tf.zeros_initializer(),
                                           kernel_regularizer=weights_initializer,
                                           bias_regularizer=None,
                                           activity_regularizer=None,
                                           trainable=True,
                                           name=name,
                                           reuse=None)

    # dropout (apply dropout after activation):
    with tf.variable_scope('fc1/'):
        # To reduce overfitting, we will apply dropout before the readout layer.
        # We create a placeholder for the probability that a neuron's output is
        # kept during dropout -> keep_prob \elem (0,1]. This allows us to turn
        # dropout on during training (keep_prob \elem (0,1)), and turn it off
        # during testing (keep_prop = 1).
        # keep_prob = tf.placeholder(tf.float32)
        # fc1_drop = tf.nn.dropout(x=fc1, keep_prob=keep_prob, noise_shape=None,
        #                          seed=None, name='fc1_dropout')
        # or alternatively use:
        # here rate is the dropout rate, between 0 and 1. E.g. "rate=0.1" would
        # drop out 10% of input units.
        fc1_drop = tf.layers.dropout(inputs=fc1, rate=dropout_rate,
                                     noise_shape=None, seed=None,
                                     training=training,
                                     name='fc1_dropout')
        nnhelp._activation_summary(fc1_drop)
    # last layer (contains predictions, do not apply a relu activation (-> use
    # linear activation instead):)
    # for the last layer we use a Xavier (uniform=True) weight initialization,
    # because we apply softmax later
    weights_initializer = tf.truncated_normal_initializer(stddev=0.1)
    # weights_initializer = tf.contrib.layers.xavier_initializer(uniform=False,
    #                                                            seed=None,
    #                                                            dtype=tf.float32)


    return
