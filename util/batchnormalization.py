import numpy as np
import tensorflow as tf
from nn_helpers import _variable_on_device

def batch_norm(x, out_cannels, phase_train, decay=0.9, scope='bn',
               device='/cpu:0'):
    """
    Batch normalization on convolutional feature maps 
    (source: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow).
    Implementation of paper Ioffe, Szegedy: 'Batch Normalization: Accelerating
    Deep Network Training by Reducing Internal Covariate Shift', 2015.
    Implements a normalization after each convolutional layer right before
    applying the non-linear activation function (ReLU). The normalizing is done
    by formula:
    
    batch_scale * (conv - mean(conv)) / std(conv + epsilon) + batch_bias
        batch_scale: parameter that scales the normalized value
        batch_bias: parameter that shifts the normalized value
        epsilon: small number to ensure that we do not divide by zero
        conv: result of applying convolutional operation in a cnn. This is the
              'input' for the batch normalization step
    
    Basically, this normalization step is applied to a convolutional layer after
    the 'convolutional step', i.e. applying function  tf.nn.conv2d(inputs,
    kernel, [1, stride, stride, 1], padding=padding). The output of this 
    function is a 4D tensor with shape 
    [batch, (in_height + 2 x P - kernel_height) / stride + 1,  
     (in_width + 2 x P - kernel_width) / stride + 1, out_channels].
    (This convolutional operation corresponds to a matmul operation in a fully
    connected nn). After the convolution (or correlation) we would add a bias
    term to the outcome, but if we apply batch normalization we ommit
    pre-batch normalization bias. The effect of this bias would be eliminated
    when subtracting the batch mean. Instead, the role of the bias is performed
    by the new batch_bias variable.

    For convolutional layers, we additionally want the normalization to obey the
    convolutional property – so that different elements of the same feature map,
    at different locations, are normalized in the same way. To achieve this, we
    jointly normalize all the activations in a mini-batch, over all locations.
    For a mini-batch of size m and feature maps of size h × w, we use the
    effective mini-batch of size m * h * w, i.e. we calculate the moments (mean,
    variance) over the first three dimensions (batch, out_height, out_width) ->
    so the mean and variance are 1 dimensional vectors with size out_channels.
    We learn a pair of parameters batch_scale and batch_bias per feature map,
    rather than per activation.
    
    Args:
        x:           Tensor, 4D BHWD input maps
        out_cannels: integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        batch-normalized maps
    """
    with tf.variable_scope(scope):
        # define/initialize the batch normalization parameters (scale and
        # offset/bias) to learn:
        batch_bias = _variable_on_device('batch_bias_', shape=[out_cannels], 
                                         initializer=tf.constant_initializer(0.0),
                                         device=device)
        batch_scale = _variable_on_device('batch_scale_', shape=[out_cannels], 
                                         initializer=tf.constant_initializer(1.0),
                                         device=device)
        # calculate the moments (mean, var) of the current mini-batch:
        # note: we also average over the spacial dimensions in cnns (output: 1D
        # tensor with length: out_channels)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        # use the moments of the current mini-batch and the previous seen
        # mini-batches to calculate an exponential moving average (ema):
        # shadow_variable = decay * shadow_variable + (1 - decay) * variable
        # with shadow_variable: value of the ema at any time step/period t
        #                       (here: running average of mean or)
        #      variable: value at a time step/period t (here: mean or variance)
        #      decay: degree of weighting decrease, a constant smoothing factor
        #             between 0 and 1. A higher decay discounts older
        #             observations faster
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            # ensure that ema_apply_op is calculated before the next operation
            # is done. (needed in parallel computation):
            with tf.control_dependencies([ema_apply_op]):
                # generate an updated batch_mean and batch_var tensor (with same
                # shape of previous tensors):
                return tf.identity(batch_mean), tf.identity(batch_var)
        
        # calculate mean and variance
        # apply condition: if phase_train is true (we are in the network
        # training stage) return  mean_var_with_update, elif phase_train is
        # false (we are in the network testing stage) return just the current
        # mini-batch mean/variance
        mean, var = tf.cond(phase_train, mean_var_with_update,
                            lambda: (ema.average(batch_mean),
                                     ema.average(batch_var)))
        # perform batch normalization:
        normed = tf.nn.batch_normalization(x, mean, var, batch_bias,
                                           batch_scale, 1e-4)
    return normed


if __name__ == "__main__":
    import math

    # define parameters:
    in_channels, out_cannels = 3, 16
    # size of kernel (ksize x ksize):
    ksize = 3
    stride = 1
    # define placeholders:
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    input_image = tf.placeholder(tf.float32, name='input_image')
    # define weights with msra weight initialization: 
    kernel = tf.Variable(tf.truncated_normal([ksize, ksize, in_channels,
                                              out_cannels], 
                                             stddev=math.sqrt(2.0/(ksize*ksize*out_cannels))),
                                             name='kernel')
    # perform convolutional step:
    conv = tf.nn.conv2d(input_image, kernel, [1,stride,stride,1],
                        padding='SAME')
    # perform batch normalization
    conv_bn = batch_norm(conv, out_cannels, phase_train)
    # apply non-linear activation
    relu = tf.nn.relu(conv_bn)

    with tf.Session() as sess:
        # initialize all variables
        sess.run(tf.initialize_all_variables())
        for i in range(20):
            # generate '20 (9 x 9) test (color) images' with batch size of 4:
            test_image = np.random.rand(4, 9, 9, 3)
            # apply relu
            sess_outputs = sess.run([relu], {input_image.name: test_image,
                                    phase_train.name: True})
            print(sess_outputs)

