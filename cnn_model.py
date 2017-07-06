 #!/usr/bin/env python

"""
Module that contains the CNN model!
"""

import sys
sys.path.append('./util')

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


__all__ = ['model_narihira2015']   


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


def create_inference_graph(save_path, device='/cpu:0'):
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
    model_out = model_narihira2015(inputs=x, training=training, device=device)

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


if __name__ == "__main__":
    create_inference_graph(save_path='logs/inference_graphs/narihira2015/',
                           device='/cpu:0')

