#!/usr/bin/env python

"""
Module that contains classes for creating file structures suitable for
deep learning neural network feeding.
"""

import numpy as np
import tensorflow as tf

__author__ = "Udo Dehm"
__copyright__ = "Copyright 2017"
__credits__ = ["Udo Dehm"]
__license__ = ""
__version__ = "0.1"
__maintainer__ = "Udo Dehm"
__email__ = "udo.dehm@mailbox.org"
__status__ = "Development" 
 

__all__ = ['_activation_summary', '_variable_on_device', 'network_params']


def _activation_summary(x):
    """
    Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
        x: Tensor
    Returns:
        nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    #     session. This helps the clarity of presentation on tensorboard.
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    # create summaries that give outputs (TensorFlow ops that output protocol
    #     buffers containing 'summarized' data) of histogram summaries, that let
    #     us take a look at shape of distributions of parameters, like
    #     weights/biases (tf.summary.histogram()).
    #     add some summaries to show distributions of 
    #     weights/biases/activations:
    tf.summary.histogram(name=x.op.name + '/activations', values=x)
    # tf.histogram_summary(tensor_name + '/activations', x)
    # summary to measure and report sparsity (tf.nn.zero_fraction() returns the 
    #     fraction of zeros in value, with type float32):
    tf.summary.scalar(name=x.op.name + '/sparsity',
                      tensor=tf.nn.zero_fraction(x))
    # tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_device(name,
                        shape,
                        initializer=tf.truncated_normal_initializer(stddev=1e-4),
                        trainable=True,
                        device='/cpu:0'):
    """
    Helper to create a Variable stored on specified device memory. This function
    uses the function tf.get_variable() to initialize variables. This function
    is prefered over tf.Variable() because it allowes Variable sharing. For
    further details see 
    https://www.tensorflow.org/programmers_guide/variable_scope.
    Args:
        name: name of the variable
        shape: list of ints, shape has to be of form:
               [kernel_height, kernel_width, in_channels, out_channels] for 
               weights or for biases: [out_channels] with out_channels=# kernels
        initializer: initializer for Variable (default: 
                     tf.truncated_normal_initializer(stddev=1e-4))
        trainable: if True, Variable is implemented as a trainable Variable
                   (default), else it is not trainable.
        device: string that specifies device to store variable, e.g. '/cpu:0' 
                for standard CPU (default, it is recomended to store variables
                on CPU if these variables are shared between several GPUs.
                Transferring data to and from GPUs is quite slow) or '/gpu:0'
                for standard GPU
    Returns: 
        Variable Tensor
    """
    # We instantiate all variables using tf.get_variable() instead of
    #     tf.Variable() in order to share variables across multiple GPU training
    #     runs. If we only ran this model on a single GPU, we could simplify
    #     this function by replacing all instances of tf.get_variable() with
    #     tf.Variable().
    with tf.device(device):
        return tf.get_variable(name, shape, initializer=initializer,
                               trainable=trainable)

def network_params():
    """
    Calculates the total number of a neural network.
    Make sure to call it after the complete training graph is built.
    :returns: Total number of network parameters
    """
    # calculate total number of network parameters:
    total_parameters = 0
    # get trainable parameters (biases, weights of cnn filters etc.):
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = int(np.prod(np.array(shape)))
        total_parameters += variable_parametes
    return total_parameters
