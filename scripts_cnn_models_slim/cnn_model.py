 #!/usr/bin/env python

"""
Module that contains the CNN model!
"""

import sys
sys.path.append('./util')

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from math import sqrt

import cnn_helpers as cnnhelp
import nn_helpers as nnhelp
import initializers_custom as ic


__author__ = "ud"
__copyright__ = "Copyright 2017"
__credits__ = ["phthalo@mailbox.org"]
__license__ = ""
__version__ = "0.1"
__maintainer__ = "ud"
__email__ = "phthalo@mailbox.org"
__status__ = "Development"


__all__ = ['model_narihira2015', 'create_inference_graph', 'upproject_arg_scope',
           '_prepare_indices', '_unpool_as_conv', 'up_project']


def model_narihira2015(inputs,
                       training=tf.placeholder(tf.bool),
                       device='/cpu:0'):
    """
    :param inputs: input to the network of form [batch, height, width, channels]
    :type inputs: tf.placeholder with corresponding shape
    :param training: indicates if network is trained (training -> True) or
        tested/validated (training -> False)
    :param training: np bool or tf bool
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
    
    f, s, p, k_in, k_out, name = [12, 4, 'SAME', 3, 96, 'conv_s1-1']
    conv_s11 = cnnhelp.conv2d_layer(inputs=inputs,
                                    kernel_shape=[f, f, k_in, k_out],
                                    stride=s,
                                    padding=p,
                                    name_conv_layer=name,
                                    weights_initializer=weights_initializer,
                                    use_bn=True,
                                    training=training,
                                    relu=True,
                                    device=device)
    print(conv_s11)

    f, s, p, name = [2, 2, 'SAME', 'max_pool_s1-1']
    pool_s11 = tf.nn.max_pool(value=conv_s11, 
                              ksize=[1, f, f, 1],
                              strides=[1, s, s, 1],
                              padding=p,
                              name=name)

    print(pool_s11)

    f, s, p, k_in, k_out, name = [5, 1, 'SAME', k_out, 256, 'conv_s1-2']
    conv_s12 = cnnhelp.conv2d_layer(inputs=pool_s11,
                                    kernel_shape=[f, f, k_in, k_out],
                                    stride=s,
                                    padding=p,
                                    name_conv_layer=name,
                                    weights_initializer=weights_initializer,
                                    use_bn=True,
                                    training=training,
                                    relu=True,
                                    device=device)
    print(conv_s12)

    f, s, p, name = [2, 2, 'SAME', 'max_pool_s1-2']
    pool_s12 = tf.nn.max_pool(value=conv_s12, 
                              ksize=[1, f, f, 1],
                              strides=[1, s, s, 1],
                              padding=p,
                              name=name)
    print(pool_s12)
    

    f, s, p, k_in, k_out, name = [3, 1, 'SAME', k_out, 256, 'conv_s1-3']
    conv_s13 = cnnhelp.conv2d_layer(inputs=pool_s12,
                                    kernel_shape=[f, f, k_in, k_out],
                                    stride=s,
                                    padding=p,
                                    name_conv_layer=name,
                                    weights_initializer=weights_initializer,
                                    use_bn=True,
                                    training=training,
                                    relu=True,
                                    device=device)
    print(conv_s13)
    

    f, s, p, k_in, k_out, name = [3, 1, 'SAME', k_out, 384, 'conv_s1-4']
    conv_s14 = cnnhelp.conv2d_layer(inputs=conv_s13,
                                    kernel_shape=[f, f, k_in, k_out],
                                    stride=s,
                                    padding=p,
                                    name_conv_layer=name,
                                    weights_initializer=weights_initializer,
                                    use_bn=True,
                                    training=training,
                                    relu=True,
                                    device=device)
    print(conv_s14)


    f, s, p, k_in, k_out, name = [3, 1, 'SAME', k_out, 256, 'conv_s1-5']
    conv_s15 = cnnhelp.conv2d_layer(inputs=conv_s14,
                                    kernel_shape=[f, f, k_in, k_out],
                                    stride=s,
                                    padding=p,
                                    name_conv_layer=name,
                                    weights_initializer=weights_initializer,
                                    use_bn=True,
                                    training=training,
                                    relu=True,
                                    device=device)
    print(conv_s15)

    f, s, p, name = [2, 2, 'SAME', 'max_pool_s1-5']
    pool_s15 = tf.nn.max_pool(value=conv_s15, 
                              ksize=[1, f, f, 1],
                              strides=[1, s, s, 1],
                              padding=p,
                              name=name)
    print(pool_s15)


    f, s, p, k_in, k_out, name = [16, 8, 'SAME', k_out, 256, 'deconv_s1']
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
                                           kernel_regularizer=None,
                                           bias_regularizer=None,
                                           activity_regularizer=None,
                                           trainable=True,
                                           name=name,
                                           reuse=None)
    print(deconv_s1)


    f, s, p, k_in, k_out, name = [1, 1, 'VALID', k_out, 64, 'conv_s1-6']
    conv_s16 = cnnhelp.conv2d_layer(inputs=deconv_s1,
                                    kernel_shape=[f, f, k_in, k_out],
                                    stride=s,
                                    padding=p,
                                    name_conv_layer=name,
                                    weights_initializer=weights_initializer,
                                    use_bn=True,
                                    training=training,
                                    relu=True,
                                    device=device)
    print(conv_s16)

    ############################################################################
    # scale 2:
    f, s, p, k_in, k_out, name = [10, 2, 'SAME', 3, 96, 'conv_s2-1']
    conv_s21 = cnnhelp.conv2d_layer(inputs=inputs,
                                    kernel_shape=[f, f, k_in, k_out],
                                    stride=s,
                                    padding=p,
                                    name_conv_layer=name,
                                    weights_initializer=weights_initializer,
                                    use_bn=True,
                                    training=training,
                                    relu=True,
                                    device=device)
    print(conv_s21)

    f, s, p, name = [2, 2, 'SAME', 'max_pool_s2-1']
    pool_s21 = tf.nn.max_pool(value=conv_s21, 
                              ksize=[1, f, f, 1],
                              strides=[1, s, s, 1],
                              padding=p,
                              name=name)
    print(pool_s21)


    concat = tf.concat(values=[pool_s21, conv_s16], axis=-1, name='concat')
    print(concat)

    
    f, s, p, k_in, k_out, name = [5, 1, 'SAME', k_out + 64, 64, 'conv_s2-2']
    conv_s22 = cnnhelp.conv2d_layer(inputs=concat,
                                    kernel_shape=[f, f, k_in, k_out],
                                    stride=s,
                                    padding=p,
                                    name_conv_layer=name,
                                    weights_initializer=weights_initializer,
                                    use_bn=True,
                                    training=training,
                                    relu=True,
                                    device=device)
    print(conv_s22)


    f, s, p, k_in, k_out, name = [5, 1, 'SAME', k_out, 64, 'conv_s2-3']
    conv_s23 = cnnhelp.conv2d_layer(inputs=conv_s22,
                                    kernel_shape=[f, f, k_in, k_out],
                                    stride=s,
                                    padding=p,
                                    name_conv_layer=name,
                                    weights_initializer=weights_initializer,
                                    use_bn=True,
                                    training=training,
                                    relu=True,
                                    device=device)
    print(conv_s23)
    

    f, s, p, k_in, k_out, name = [5, 1, 'SAME', k_out, 64, 'conv_s2-4']
    conv_s24 = cnnhelp.conv2d_layer(inputs=conv_s23,
                                    kernel_shape=[f, f, k_in, k_out],
                                    stride=s,
                                    padding=p,
                                    name_conv_layer=name,
                                    weights_initializer=weights_initializer,
                                    use_bn=True,
                                    training=training,
                                    relu=True,
                                    device=device)
    print(conv_s24)


    # albedo:
    f, s, p, k_in, k_out, name = [5, 1, 'SAME', k_out, 64, 'conv_s2-5_albedo']
    conv_s25_albedo = cnnhelp.conv2d_layer(inputs=conv_s24,
                                           kernel_shape=[f, f, k_in, k_out],
                                           stride=s,
                                           padding=p,
                                           name_conv_layer=name,
                                           weights_initializer=weights_initializer,
                                           use_bn=True,
                                           training=training,
                                           relu=True,
                                           device=device)
    print(conv_s25_albedo)

    f, s, p, k_in, k_out, name = [8, 4, 'SAME', k_out, 3, 'deconv_s2out_albedo']
    deconv_s2_albedo = tf.layers.conv2d_transpose(inputs=conv_s25_albedo,
                                                  filters=k_out,
                                                  kernel_size=f,
                                                  strides=s,
                                                  padding=p,
                                                  data_format='channels_last',
                                                  activation=None,
                                                  use_bias=True,
                                                  kernel_initializer=None,
                                                  bias_initializer=tf.zeros_initializer(),
                                                  kernel_regularizer=None,
                                                  bias_regularizer=None,
                                                  activity_regularizer=None,
                                                  trainable=True,
                                                  name=name,
                                                  reuse=None)
    print(deconv_s2_albedo)


    # shading:
    f, s, p, k_in, k_out, name = [5, 1, 'SAME', 64, 64, 'conv_s2-5_shading']
    conv_s25_shading = cnnhelp.conv2d_layer(inputs=conv_s24,
                                            kernel_shape=[f, f, k_in, k_out],
                                            stride=s,
                                            padding=p,
                                            name_conv_layer=name,
                                            weights_initializer=weights_initializer,
                                            use_bn=True,
                                            training=training,
                                            relu=True,
                                            device=device)
    print(conv_s25_shading)

    f, s, p, k_in, k_out, name = [8, 4, 'SAME', k_out, 3, 'deconv_s2out_shading']
    deconv_s2_shading = tf.layers.conv2d_transpose(inputs=conv_s25_shading,
                                                   filters=k_out,
                                                   kernel_size=f,
                                                   strides=s,
                                                   padding=p,
                                                   data_format='channels_last',
                                                   activation=None,
                                                   use_bias=True,
                                                   kernel_initializer=None,
                                                   bias_initializer=tf.zeros_initializer(),
                                                   kernel_regularizer=None,
                                                   bias_regularizer=None,
                                                   activity_regularizer=None,
                                                   trainable=True,
                                                   name=name,
                                                   reuse=None)
    print(deconv_s2_shading)
    return deconv_s2_albedo, deconv_s2_shading


def create_inference_graph(modelfunc, save_path, device='/cpu:0'):
    """
    Create an inference graph (defined above): A complete neural network that
    (ONLY) can do inference (forward pass). 
    It contains a graph from input to output of the network.
    It does not contain a loss function and optimization ops (which are
    necessary for backpropagation / learning).
    These operations have to be included after loading the model.

    Until now, the model which is created is hard coded.
    ATTENTION: be careful creating model from jupyter file because
    of saving properties!
    to be sure it would be best to save models from a python script (update: if
    kernel is restarted, it works quite fine in jupyter)
    :param save_path: path where the tensorflow (inference) graph should be
        saved
    :type device: str
    :param device: device where to save variables: e.g. standard cpu: '/cpu:0',
        standard gpu: '/gpu:0' (default: '/cpu:0')
    :type device: str
    :return: nothing
    """
    # make sure we start with a new, empty graph before building it:
    tf.reset_default_graph()

    # setup input placeholders with the shape=None (we do not want to save fixed
    # image sizes because we might enter different image dimenstions into the
    # network. But ATTENTION: we need to be careful which dimensions we add to
    # the network because the network (first convolution layer can only handle
    # certain dimensions):
    x = tf.placeholder(dtype=tf.float32, shape=None, name='input') 
    # We want to create summaries that give us outputs (TensorFlow ops that
    # output protocol buffers containing 'summarized' data) of images, e.g. to
    # check right formating of input images (tf.summary.images()) display some
    # input data in tensorboard under summary image:
    # (images are built from tensor which must be 4-D with shape
    # [batch_size, height, width, channels] where
    # channels \elem {1 ->grayscale, 3 -> rgb, 4 -> rgba}. 
    # max_outputs: Max number of batch elements to generate images for)
    tf.summary.image(name='input', tensor=x, max_outputs=3, collections=None)

    # bool variable that indicates if we are in training mode (training=True) or
    # valid/test mode (training=False) this indicator is important if dropout
    # or/and batch normalization is used.
    training = tf.placeholder(tf.bool, name='is_training')

    # define model in a separate function:
    model_out = modelfunc(inputs=x, training=training, device=device)

    # to get every summary defined above we merge them to get one target:
    merge_train_summaries = tf.summary.merge_all()

    # define a FileWriter op which writes summaries defined above to disk:
    summary_writer = tf.summary.FileWriter(save_path)

    # Op that initializes global variables in the graph:
    init_global = tf.global_variables_initializer()

    # Create a saver.
    saver = tf.train.Saver()
    # Generates MetaGraphDef (this is the important saving operation. it saves
    # the inference graph!):
    saver.export_meta_graph(save_path + 'tfmodel_inference.meta')

    with tf.Session() as sess:
        # initialize all variables:
        sess.run([init_global])
        # Adds a Graph to the event file. 
        # create summary that give output (TensorFlow op that output protocol
        # buffers containing 'summarized' data) of the built Tensorflow graph:
        summary_writer.add_graph(sess.graph)
                                                       
        # # Runs to network output:
        # feed_dict = {x: np.random.random([1] + IMAGE_SHAPE), training: False}
        # model_output = sess.run(model_out, feed_dict=feed_dict)
        # s = sess.run(merge_train_summaries, feed_dict=feed_dict)
        # summary_writer.add_summary(summary=s)

        # # this saves data:
        # saver.save(sess, save_path + 'tfmodel_inference')
        # return model_output


################################################################################
# Operations, specific to FCRN (UpProjection)
# see https://arxiv.org/abs/1606.00373
# unpool layers performed with conv2d layers (more efficient than unpooling
# layers)
################################################################################


def upproject_arg_scope(weight_decay=0.0001,
                        batch_norm_decay=0.997,
                        batch_norm_epsilon=1e-5,
                        batch_norm_scale=True,
                        activation_fn=tf.nn.relu,
                        use_batch_norm=True,
                        is_training=False):
    """
    Defines the default ResNet arg scope.
    Apply by using: with slim.arg_scope(upproject_arg_scope()):

    TODO(gpapan): The batch-normalization related default values above are
    appropriate for use in conjunction with the reference ResNet models
    released at https://github.com/KaimingHe/deep-residual-networks. When
    training ResNets from scratch, they might need to be tuned.

    :param weight_decay: The weight decay to use for regularizing the model.
    :param batch_norm_decay: The moving average decay when estimating layer
     activation statistics in batch normalization.
    :param batch_norm_epsilon: Small constant to prevent division by zero when
     normalizing activations by their variance in batch normalization.
    :param batch_norm_scale: If True, uses an explicit `gamma` multiplier to
     scale the activations in the batch normalization layer.
    :param activation_fn: The activation function which is used in ResNet.
    :param use_batch_norm: Whether or not to use batch normalization.

    :returns: An 'arg_scope' to use for the upproject models.
    """

    batch_norm_params = {'decay': batch_norm_decay,
                         'epsilon': batch_norm_epsilon,
                         'scale': batch_norm_scale,
                         'updates_collections': tf.GraphKeys.UPDATE_OPS,
                         'is_training': is_training}

    with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=slim.variance_scaling_initializer(),
                        activation_fn=activation_fn,
                        normalizer_fn=slim.batch_norm if use_batch_norm else None,
                        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc


def _prepare_indices(before, row, col, after, dims):
    """

    :param before:
    :param row:
    :param col:
    :param after:
    :param dims:
    :return:
    """
    x0, x1, x2, x3 = np.meshgrid(before, row, col, after)

    x_0 = tf.Variable(x0.reshape([-1]), name='x_0', trainable=False)
    x_1 = tf.Variable(x1.reshape([-1]), name='x_1', trainable=False)
    x_2 = tf.Variable(x2.reshape([-1]), name='x_2', trainable=False)
    x_3 = tf.Variable(x3.reshape([-1]), name='x_3', trainable=False)

    summand2 = dims[3].value * x_2
    summand3 = 2 * dims[2].value * dims[3].value * x_0 * 2 * dims[1].value
    summand4 = 2 * dims[2].value * dims[3].value * x_1
    linear_indices = x_3 + summand2 + summand3 + summand4

    linear_indices_int = tf.to_int32(linear_indices)

    return linear_indices_int


def _unpool_as_conv(input_data,
                    num_outputs,
                    batch_size,
                    stride=1,
                    relu=False,
                    use_batch_norm=True,
                    bn_decay=0.999,
                    bn_epsilon=0.001,
                    is_training=True):
    """
    Model upconvolutions (unpooling + convolution) as interleaving feature
    maps of four convolutions (A,B,C,D). Building block for up-projections.

    :param input_data:
    :param num_outputs:
    :param stride:
    :param ReLU:
    :param BN:
    :param is_training:
    :return:
    """
    with tf.variable_scope(name_or_scope='unpool_convs',
                           default_name='unpool_convs',
                           values=[input_data]):
        with slim.arg_scope(upproject_arg_scope(weight_decay=0.0001,
                                                batch_norm_decay=bn_decay,
                                                batch_norm_epsilon=bn_epsilon,
                                                batch_norm_scale=True,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                is_training=is_training)):
            ####################################################################
            # Convolution A (3x3)
            outputA = slim.conv2d(inputs=input_data,
                                  num_outputs=num_outputs,
                                  kernel_size=[3, 3],
                                  padding='SAME',
                                  stride=stride,
                                  scope='convA')
            ####################################################################
            # Convolution B (2x3)
            # add zeros to input data: 1 row on top and one row on left and
            # right of image respectively
            padded_input_B = tf.pad(input_data,
                                    [[0, 0], [1, 0], [1, 1], [0, 0]],
                                    'CONSTANT')
            outputB = slim.conv2d(inputs=padded_input_B,
                                  num_outputs=num_outputs,
                                  kernel_size=[2, 3],
                                  padding='VALID',
                                  stride=stride,
                                  scope='convB')
            ####################################################################
            # Convolution C (3x2)
            padded_input_C = tf.pad(input_data,
                                    [[0, 0], [1, 1], [1, 0], [0, 0]],
                                    'CONSTANT')
            outputC = slim.conv2d(inputs=padded_input_C,
                                  num_outputs=num_outputs,
                                  kernel_size=[3, 2],
                                  padding='VALID',
                                  stride=stride,
                                  scope='convC')
            ####################################################################
            # Convolution D (2x2)
            padded_input_D = tf.pad(input_data,
                                    [[0, 0], [1, 0], [1, 0], [0, 0]],
                                    'CONSTANT')
            outputD = slim.conv2d(inputs=padded_input_D,
                                  num_outputs=num_outputs,
                                  kernel_size=[2, 2],
                                  padding='VALID',
                                  stride=stride,
                                  scope='convD')

    ############################################################################
    # Interleaving elements of the four feature maps:
    with tf.variable_scope('interleaving_elements'):
        dims = outputA.get_shape()
        # new spatial dimensions:
        dim1 = dims[1] * 2
        dim2 = dims[2] * 2

        A_row_indices = range(0, dim1, 2)
        A_col_indices = range(0, dim2, 2)
        B_row_indices = range(1, dim1, 2)
        B_col_indices = range(0, dim2, 2)  # = A_col_indices
        C_row_indices = range(0, dim1, 2)  # = A_row_indices
        C_col_indices = range(1, dim2, 2)
        D_row_indices = range(1, dim1, 2)  # = B_row_indices
        D_col_indices = range(1, dim2, 2)  # = C_col_indices

        all_indices_before = range(int(batch_size))
        all_indices_after = range(dims[3])

        A_linear_indices = _prepare_indices(all_indices_before,
                                            A_row_indices,
                                            A_col_indices,
                                            all_indices_after,
                                            dims)
        B_linear_indices = _prepare_indices(all_indices_before,
                                            B_row_indices,
                                            B_col_indices,
                                            all_indices_after,
                                            dims)
        C_linear_indices = _prepare_indices(all_indices_before,
                                            C_row_indices,
                                            C_col_indices,
                                            all_indices_after,
                                            dims)
        D_linear_indices = _prepare_indices(all_indices_before,
                                            D_row_indices,
                                            D_col_indices,
                                            all_indices_after,
                                            dims)

        A_flat = tf.reshape(tf.transpose(outputA, [1, 0, 2, 3]), [-1])
        B_flat = tf.reshape(tf.transpose(outputB, [1, 0, 2, 3]), [-1])
        C_flat = tf.reshape(tf.transpose(outputC, [1, 0, 2, 3]), [-1])
        D_flat = tf.reshape(tf.transpose(outputD, [1, 0, 2, 3]), [-1])

        Y_flat = tf.dynamic_stitch([A_linear_indices,
                                    B_linear_indices,
                                    C_linear_indices,
                                    D_linear_indices],
                                   [A_flat, B_flat,
                                    C_flat, D_flat])
        Y = tf.reshape(Y_flat, shape=tf.to_int32([batch_size, dim1.value,
                                                  dim2.value,
                                                  dims[3].value]))

    ############################################################################
    if use_batch_norm:
        Y = slim.batch_norm(inputs=Y,
                            decay=bn_decay,
                            scale=True,
                            epsilon=bn_epsilon,
                            activation_fn=None,
                            is_training=is_training,
                            scope='batch_norm')
    ############################################################################
    if relu:
        pass
        Y = tf.nn.relu(Y, name='relu')

    return Y

def up_project(input_data,
               kernel_size,
               num_outputs,
               batch_size,
               scope,
               stride=1,
               use_batch_norm=False,
               bn_decay=0.999,
               bn_epsilon=0.001,
               is_training=True):
    """
    tf slim implementation of up-projection layers of publication
    https://arxiv.org/abs/1606.00373 that leads to 2 * spatial input shape.

    :param input_data: tf input tensor node
    :param num_outputs: number of output filters (output channels).
    :type num_outputs: int
    :param kernel_size: A sequence of 2 positive integers specifying the spatial
     dimensions of the filters ['width', 'height'].
    :type kernel_size: list
    :param batch_size:
    :param scope:
    :param stride:
    :param BN:
    """
    # Create residual upsampling layer (UpProjection)
    with tf.variable_scope(name_or_scope=scope,
                           default_name='up_projection',
                           values=[input_data]):
        # Branch 1
        with tf.variable_scope(name_or_scope='branch1', values=[input_data]):
             # Interleaving Convs of 1st branch
             out = _unpool_as_conv(input_data=input_data,
                                   num_outputs=num_outputs,
                                   batch_size=batch_size,
                                   stride=stride,
                                   relu=True,
                                   use_batch_norm=True,
                                   bn_decay=bn_decay,
                                   bn_epsilon=bn_epsilon,
                                   is_training=is_training)
             with slim.arg_scope(upproject_arg_scope(weight_decay=0.0001,
                                                     batch_norm_decay=bn_decay,
                                                     batch_norm_epsilon=bn_epsilon,
                                                     batch_norm_scale=True,
                                                     activation_fn=None,
                                                     use_batch_norm=use_batch_norm,
                                                     is_training=is_training)):
                 # Convolution following the upProjection on the 1st branch
                 branch1_output = slim.conv2d(inputs=out,
                                              num_outputs=num_outputs,
                                              kernel_size=kernel_size,
                                              padding='SAME',
                                              stride=stride,
                                              scope='conv')

        ########################################################################
        # Branch 2
        with tf.variable_scope(name_or_scope='branch2', values=[input_data]):
             # Interleaving convolutions and output of 2nd branch
             branch2_output = _unpool_as_conv(input_data=input_data,
                                              num_outputs=num_outputs,
                                              batch_size=batch_size,
                                              stride=stride,
                                              relu=False,
                                              use_batch_norm=True,
                                              bn_decay=bn_decay,
                                              bn_epsilon=bn_epsilon,
                                              is_training=is_training)

        # sum branches
        output = tf.add_n([branch1_output, branch2_output],
                          name='sum_branches')
        # ReLU
        output = tf.nn.relu(output, name='relu')
        return output


def _unpool_as_conv_udo(input_data,
                        num_outputs,
                        stride=1,
                        relu=False,
                        use_batch_norm=True,
                        bn_decay=0.999,
                        bn_epsilon=0.001,
                        is_training=True):
    """
    Model upconvolutions (unpooling + convolution) as interleaving feature
    maps of four convolutions (A,B,C,D). Building block for up-projections.

    :param input_data:
    :param num_outputs:
    :param stride:
    :param ReLU:
    :param BN:
    :param is_training:
    :return:
    """
    with tf.variable_scope(name_or_scope='unpool_convs',
                           default_name='unpool_convs',
                           values=[input_data]):
        with slim.arg_scope(upproject_arg_scope(weight_decay=0.0001,
                                                batch_norm_decay=bn_decay,
                                                batch_norm_epsilon=bn_epsilon,
                                                batch_norm_scale=True,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                is_training=is_training)):
            ####################################################################
            # Convolution A (3x3)
            outputA = slim.conv2d(inputs=input_data,
                                  num_outputs=num_outputs,
                                  kernel_size=[3, 3],
                                  padding='SAME',
                                  stride=stride,
                                  scope='convA')
            ####################################################################
            # Convolution B (2x3)
            # add zeros to input data: 1 row on top and one row on left and
            # right of image respectively
            padded_input_B = tf.pad(input_data,
                                    [[0, 0], [1, 0], [1, 1], [0, 0]],
                                    'CONSTANT')
            outputB = slim.conv2d(inputs=padded_input_B,
                                  num_outputs=num_outputs,
                                  kernel_size=[2, 3],
                                  padding='VALID',
                                  stride=stride,
                                  scope='convB')
            ####################################################################
            # Convolution C (3x2)
            padded_input_C = tf.pad(input_data,
                                    [[0, 0], [1, 1], [1, 0], [0, 0]],
                                    'CONSTANT')
            outputC = slim.conv2d(inputs=padded_input_C,
                                  num_outputs=num_outputs,
                                  kernel_size=[3, 2],
                                  padding='VALID',
                                  stride=stride,
                                  scope='convC')
            ####################################################################
            # Convolution D (2x2)
            padded_input_D = tf.pad(input_data,
                                    [[0, 0], [1, 0], [1, 0], [0, 0]],
                                    'CONSTANT')
            outputD = slim.conv2d(inputs=padded_input_D,
                                  num_outputs=num_outputs,
                                  kernel_size=[2, 2],
                                  padding='VALID',
                                  stride=stride,
                                  scope='convD')

    return outputA, outputB, outputC, outputD

if __name__ == "__main__":
    create_inference_graph(modelfunc=model_narihira2015, 
                           save_path='models/narihira2015/',
                           device='/cpu:0')

