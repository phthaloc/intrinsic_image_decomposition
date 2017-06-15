#!/usr/bin/env python
   
"""
Module that contains layers for creating CNNs.
"""
             
import tensorflow as tf
import nn_helpers as nnhelp

__author__ = "Udo Dehm"
__copyright__ = "Copyright 2017"
__credits__ = ["Udo Dehm"]
__license__ = ""
__version__ = "0.1"
__maintainer__ = "Udo Dehm"
__email__ = "udo.dehm@mailbox.org"
__status__ = "Development"


__all__ = ['conv2d_layer'] 


def conv2d_layer(inputs,
                 kernel_shape,
                 stride,
                 padding='SAME',
                 name_conv_layer='conv', 
                 weights_initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                   stddev=1e-4),
                 use_bn=True,
                 training=True,
                 relu=True,
                 device='/cpu:0'):
    """
    Creates a convolutional layer with ReLU activation function with 
    (kernel_height x kernel_width x in_channels + 1) x out_channels parameters
    (weights and biases)
    Args:
        inputs: input (images) of shape
            [batch, in_height, in_width, in_channels]
        kernel_shape: kernel/ filter (weights) of shape:
            [kernel_height, kernel_width, in_channels, out_channels],
            with kernel_height=kernel_width
        stride: sliding stride for each dimension of input: 
            [stride_batch=1, stride_height, stride_width, stride_in_channels=1]
            with stride_height=stride_width=stride (int)
        padding: zero padding: VALID: no padding, SAME (default): with padding
            (tf detects automatically how many zeros have to be padded) (str)
            padding=='SAME': padding == "SAME": output_spatial_shape[i] =
            ceil(input_spatial_shape[i] / strides[i])
            padding=='VALID': output_spatial_shape[i] =
            ceil((input_spatial_shape[i] - (spatial_filter_shape[i]-1) *
            dilation_rate[i]) / strides[i])
            for more details see:
            https://www.tensorflow.org/api_docs/python/tf/nn/convolution
        name_conv_layer: name of convolutional (or input/output) layer (str,
            default: 'hidden')
        weights_initializer: initializer for weight variable (NOTICE: It is
            always legitimate to set an initializer op for weights or biases,
            even if we want to restore a model (previously saved weights and
            biases). If we do not initialize (eg. by
            tf.global_variables_initializer()) these parameters we can set them
            to the restored values!)
        use_bn: bool. True if batch normalization should be used, False if batch
            normalization is ignored (default: True)
        training: tf bool, if true training phase, if false validation/testing
            phase. This is important for using batch normalization (default =
            True)
        relu: if True, a relu activation function is applied to the weighted
            sum, if False, only a weighted sum of the inputs is given as output
            (boolean, default: True)
        device: string that specifies device to store variable, e.g. '/cpu:0'
            for standard CPU (default, it is recomended to store variables on
            CPU if these variables are shared between several GPUs. Transferring
            data to and from GPUs is quite slow) or '/gpu:0' for standard GPU
    Returns:
        tf output tensor with shape 
            [batch, (in_height + 2 x P - kernel_height) / stride + 1, 
             (in_width + 2 x P - kernel_width) / stride + 1, out_channels]
        with padding P (P=1 for)
    """
    with tf.variable_scope(name_conv_layer):
        # (NOTICE: It is always legitimate to set an initializer op for weights
        #     or biases, even if we want to restore a model (previously saved
        #     weights and biases). If we do not initialize (eg. by 
        #     tf.global_variables_initializer()) these parameters we can set
        #     them to the restored values!)
        kernel = nnhelp._variable_on_device('weights', shape=kernel_shape,
                                          initializer=weights_initializer,
                                          device=device)
#         # to tf.image_summary format [batch_size, height, width, channels]
#         kernel_transposed = tf.transpose(kernel, [3, 0, 1, 2])
#         # this will display random 3 filters from the 64 in conv1
#         tf.summary.image(name='weights', tensor=kernel_transposed, max_outputs=3)
        biases = nnhelp._variable_on_device('biases', shape=[kernel_shape[3]],
                                          initializer=tf.constant_initializer(0.1),
                                          device=device)
        # conv2d:
        #     inputs of shape [batch, in_height, in_width, in_channels]
        #     kernel (weights) of shape:
        #     [kernel_height, kernel_width, in_channels, out_channels]
        #     strides: sliding stride for each dimension of input: 
        #     [stride_batch=1, stride_height, stride_width, stride_in_channels=1]
        #     with stride_height=stride_width
        #     padding: VALID: no padding, SAME: with padding
        #     output: A Tensor. Has the same type as input, here 4D.
        #     [batch, (in_height + 2 x P - kernel_height) / stride + 1, 
        #      (in_width + 2 x P - kernel_width) / stride + 1, out_channels]
        conv = tf.nn.conv2d(inputs, kernel, [1, stride, stride, 1],
                            padding=padding)
        # using batch normalization:
        # here we do not use reuse right now, if you want to use it see also
        # here:
        # https://stackoverflow.com/documentation/tensorflow/7909/using-batch-normalization#t=201611300538141458755
        if use_bn:
            conv = tf.layers.batch_normalization(inputs=conv,  # Tensor input
                                                 axis=-1,  # axis that should be
                                                 # normalized (typically the 
                                                 # features axis). For instance,
                                                 # after a conv2d layer with
                                                 # data_format="channels_first",
                                                 # set axis=1 in
                                                 # Batch Normalization
                                                 momentum=0.99,  # Momentum for
                                                 # the moving average
                                                 epsilon=0.001,  # Small float
                                                 # added to variance to avoid
                                                 # dividing by zero
                                                 center=True,  # If True, add
                                                 # offset of beta to normalized
                                                 # tensor. If False, beta is
                                                 # ignored.
                                                 scale=True,  # If True, multiply
                                                 # by gamma. If False, gamma is
                                                 # not used. When the next layer 
                                                 # is linear (also e.g. nn.relu),
                                                 # this can be disabled since the
                                                 # scaling can be done by the
                                                 # next layer.
                                                 beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(),
                                                 beta_regularizer=None,
                                                 gamma_regularizer=None,
                                                 training=training,  # TensorFlow
                                                 # boolean scalar tensor (e.g. a
                                                 # placeholder). Whether to
                                                 # return the output in training
                                                 # mode (normalized with
                                                 # statistics of the current
                                                 # batch) or in inference mode
                                                 # (normalized with moving
                                                 # statistics).
                                                 trainable=True,
                                                 name='bn',
                                                 reuse=None  # Boolean, whether
                                                 # to reuse the weights of a
                                                 # previous layer by the same
                                                 # name.
                                                 )

        if relu:
            act = tf.nn.relu(conv + biases)
        else:
            act = conv + biases
        # write activiation within this scope to get it in the same scope name
        #     category:
        nnhelp._activation_summary(act)
    # create summaries that give outputs (TensorFlow ops that output
    #     protocol buffers containing 'summarized' data) of histogram
    #     summaries, that let us take a look at shape of distributions of
    #     parameters, like weights/biases (tf.summary.histogram()).
    #     add some summaries to show distributions of
    #     weights/biases/activations:
    tf.summary.histogram(name=kernel.op.name, values=kernel)
    tf.summary.histogram(name=biases.op.name, values=biases)
    return act, kernel

