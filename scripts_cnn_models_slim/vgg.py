# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains model definitions for versions of the Oxford VGG network.

These model definitions were introduced in the following technical report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/

Usage:
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_a(inputs)

  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_16(inputs)

@@vgg_a
@@vgg_16
@@vgg_19
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def vgg_arg_scope(weight_decay=0.0005):
  """Defines the VGG arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc


def vgg_a(inputs,
          num_classes=1000,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='vgg_a',
          fc_conv_padding='VALID'):
  """Oxford Net VGG 11-Layers version A Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.

  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'vgg_a', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 1, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout7')
      net = slim.conv2d(net, num_classes, [1, 1],
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='fc8')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if spatial_squeeze:
        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points
vgg_a.default_image_size = 224


def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID'):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If num_calsses=None, then graph is
    built without fully connected layers.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.

  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(name_or_scope=scope, default_name='vgg_16',
                         values=[inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs=inputs, repetitions=2,
                        layer=slim.conv2d, num_outputs=64, kernel_size=[3, 3],
                        scope='conv1') # default: stride=1, padding='SAME'
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      if num_classes:
          # Use conv2d instead of fully_connected layers.
          net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding,
                            scope='fc6')
          net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                             scope='dropout6')
          net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
          net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                             scope='dropout7')
          net = slim.conv2d(net, num_classes, [1, 1],
                            activation_fn=None,
                            normalizer_fn=None,
                            scope='fc8')
          # Convert end_points_collection into a end_point dict.
          end_points = slim.utils.convert_collection_to_dict(end_points_collection)
          if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
            end_points[sc.name + '/fc8'] = net
      else:
          # Convert end_points_collection into a end_point dict.
          end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      return net, end_points
vgg_16.default_image_size = 224


def vgg_16_custom(inputs, net, end_points=None, is_2scales=True, reduced=False):
    # TODO: this function is user defined. write it to a separate script!!!!
    """
    :param inputs:
    :param net:
    :param reduced: if true it reduces number of parameters in network by
    inserting a convolutional layer before the first upscaling deconvolution
    layer
    :return:
    """
    weights_initializer = tf.contrib.layers.xavier_initializer(uniform=False,
                                                               seed=None,
                                                               dtype=tf.float32)

    with tf.variable_scope(name_or_scope='scale1', default_name='scale1',
                           values=[inputs, net]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d,
                             slim.conv2d_transpose],
                            outputs_collections=end_points_collection) as asc:
            if reduced:
                f, s, p, k_in, k_out, name = [1, 1, 'VALID', 512, 128, 'conv6']
                net = slim.conv2d(inputs=net, num_outputs=k_out,
                                  kernel_size=[f, f],
                                  padding=p, scope=name,
                                  weights_initializer=weights_initializer)
                k_out_new = 128
            else:
                k_out = 512
                k_out_new = 256
            f, s, p, k_in, k_out, name = [16, 8, 'SAME', k_out, k_out_new,
                                          'deconv6']
            net = slim.conv2d_transpose(inputs=net, num_outputs=k_out,
                                        kernel_size=[f, f], stride=s, padding=p,
                                        data_format='NHWC', trainable=True,
                                        scope=name)

            f, s, p, k_in, k_out, name = [1, 1, 'VALID', k_out, 64, 'conv7']
            net = slim.conv2d(inputs=net, num_outputs=k_out, kernel_size=[f, f],
                              padding=p, scope=name,
                              weights_initializer=weights_initializer)
            if end_points:
                # append (at end) of existing ordered dict:
                end_points.update(slim.utils.convert_collection_to_dict(end_points_collection))
            else:
                # create new dict:
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)


    with tf.variable_scope(name_or_scope='scale2', default_name='scale2',
                           values=[inputs, net]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d,
                             slim.conv2d_transpose],
                            outputs_collections=end_points_collection):

            ####################################################################
            # scale 2:
            if is_2scales:
                f, s, p, k_in, k_out, name = [10, 2, 'SAME', 3, 96, 'conv1_s2']
                net_s2 = slim.conv2d(inputs=inputs, num_outputs=k_out,
                                     kernel_size=[f, f], padding=p, stride=s,
                                     scope=name,
                                     weights_initializer=weights_initializer)

                f, s, p, name = [2, 2, 'SAME', 'pool1_s2']
                net_s2 = slim.max_pool2d(inputs=net_s2, kernel_size=[f, f],
                                         stride=s, padding=p, scope=name)

                net = tf.concat(values=[net_s2, net], axis=-1, name='concat')

                f, s, p, k_in, k_out, name = [5, 1, 'SAME', k_out + 64, 64, 'conv2_s2']
            else:
                f, s, p, k_in, k_out, name = [5, 1, 'SAME', 64, 64, 'conv2_s2']
    
            net = slim.conv2d(inputs=net, num_outputs=k_out, kernel_size=[f, f],
                              padding=p, stride=s, scope=name,
                              weights_initializer=weights_initializer)

            f, s, p, k_in, k_out, name = [5, 1, 'SAME', k_out, 64, 'conv3_s2']
            net = slim.conv2d(inputs=net, num_outputs=k_out, kernel_size=[f, f],
                              padding=p, stride=s, scope=name,
                              weights_initializer=weights_initializer)

            f, s, p, k_in, k_out, name = [5, 1, 'SAME', k_out, 64, 'conv4_s2']
            net = slim.conv2d(inputs=net, num_outputs=k_out, kernel_size=[f, f],
                              padding=p, stride=s, scope=name,
                              weights_initializer=weights_initializer)

            # albedo:
            f, s, p, k_in, k_out, name = [5, 1, 'SAME', k_out, 64, 'conv5_s2_albedo']
            net_albedo = slim.conv2d(inputs=net, num_outputs=k_out,
                                     kernel_size=[f, f], padding=p, stride=s,
                                     scope=name,
                                     weights_initializer=weights_initializer)

            f, s, p, k_in, k_out, name = [8, 4, 'SAME', k_out, 3, 'deconv6_s2_albedo']
            net_albedo = slim.conv2d_transpose(inputs=net_albedo,
                                               num_outputs=k_out,
                                               kernel_size=[f, f],
                                               stride=s, padding=p,
                                               data_format='NHWC',
                                               trainable=True, scope=name,
                                               activation_fn=None)

            # shading:
            f, s, p, k_in, k_out, name = [5, 1, 'SAME', 64, 64, 'conv5_s2_shading']
            net_shading = slim.conv2d(inputs=net, num_outputs=k_out,
                                      kernel_size=[f, f], padding=p, stride=s,
                                      scope=name,
                                      weights_initializer=weights_initializer)

            f, s, p, k_in, k_out, name = [8, 4, 'SAME', k_out, 3, 'deconv6_s2_shading']
            net_shading = slim.conv2d_transpose(inputs=net_shading,
                                                num_outputs=k_out,
                                                kernel_size=[f, f],
                                                stride=s, padding=p,
                                                data_format='NHWC',
                                                trainable=True, scope=name,
                                                activation_fn=None)
            # Convert end_points_collection into a end_point dict.
            end_points.update(slim.utils.convert_collection_to_dict(end_points_collection))
            return net_albedo, net_shading, end_points


def vgg_19(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19',
           fc_conv_padding='VALID'):
  """Oxford Net VGG 19-Layers version E Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.


  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'vgg_19', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout7')
      net = slim.conv2d(net, num_classes, [1, 1],
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='fc8')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if spatial_squeeze:
        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points
vgg_19.default_image_size = 224

# Alias
vgg_d = vgg_16
vgg_e = vgg_19
