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


__all__ = ['conv2d_layer', 'get_valid_pixels', 'berhu_loss', 'l2_loss',
           'loss_fct']


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
    return act


# def loss_fct(label_albedo,
#              label_shading,
#              prediction_albedo,
#              prediction_shading,
#              lambda_):
#     """
#     Computes loss function (it compares ground truth (labels) to predictions
#     y)
#     """
#     with tf.name_scope('loss'):
#         mse_alb = tf.reduce_mean(tf.square(prediction_albedo - label_albedo))
#         reduced_alb = tf.square(tf.reduce_mean(prediction_albedo - label_albedo))
#
#         mse_shad = tf.reduce_mean(tf.square(prediction_shading - label_shading))
#         reduced_shad = tf.square(tf.reduce_mean(prediction_shading - label_shading))
#
#         # least square loss if lambda_ = 0, scale invariant loss if
#         # lambda_ = 1, average of both if lambda_ = 0.5:
#         loss = (mse_alb - lambda_ * reduced_alb) + (mse_shad - lambda_ * reduced_shad)
#         # create summaries that give outputs (TensorFlow ops that output
#         # protocol buffers containing 'summarized' data) of some scalar
#         # parameters, like evolution of loss function (how does it change
#         # over time? / rather not weights  because there are usually several
#         # of them) etc (tf.summary.scalar())
#         # (have a look at parameters that change over time (= training
#         # steps))
#         tf.summary.scalar(name='loss', tensor=loss)
#     return loss


def get_valid_pixels(image, invalid_mask=None):
    """
    :param image:
    :param invalid_mask:
    """
    # generate a mask with {0, 1} (if label>0: 1, else 0)
    # mask that decodes positive/negative values in labels (should be always in
    # [0,1] range):
    # (tf.where: Return the elements, either from x or y, depending on the
    # condition)
    valid = tf.where(condition=tf.greater_equal(image, 0.),
                     x=tf.ones_like(image),
                     y=tf.zeros_like(image))
    if invalid_mask is not None:
        # in the invalid masks mark pixels that have values >0 with 0 (invalid
        # pixels) and pixels that have values==0 with 1 (valid pixels):
        valid = tf.where(condition=tf.greater(invalid_mask, 0.),
                         x=tf.zeros_like(invalid_mask),
                         y=valid)
    return valid


def berhu_loss(label, prediction, valid_mask=None):
    """
    :param label: ground truth label image
    :type label: np.array or tf tensor RGB image
    :param prediction: prediction (network output) of label image
    :type prediction: np.array or tf tensor RGB image
    :param valid_mask: binary map with 0 for invalid pixels that are not
        considered calculating the loss and 1 for valid pixels (default: None)
    :type valid_mask: np.array or tf tensor image
    :param log: PARAMETER DELETED calculate loss in log space (take elementwise
        natural log of label and prediction) (default: True)
    :type log: bool
    """
#     if log:
#         # define offset to ensure that there will be no log(0)=-inf:
#         offset = 0.5
#         # natural logarithm (base e):
#         # take abs value to ensure that there will be no log(-x):
#         label = tf.log(tf.abs(label) + offset)
#         prediction = tf.log(tf.abs(prediction) + offset)

    # Calculate absolute difference between prediction and label:
    diff = tf.abs(prediction - label)
    # get limit where L1 / L2 loss will be applied:
    lim = 0.2 * tf.reduce_max(diff)
    # apply L2 loss to regions where difference is > lim,
    # else (regions with absolute diff <= lim) apply L1:
    loss = tf.where(condition=tf.greater(diff, lim),
                    x=0.5 * ((diff ** 2 / lim) + lim),
                    y=diff)

    if valid_mask is not None:
        # get rid of invalid pixels (which do not contribute to loss):
        loss_valid = tf.multiply(valid_mask, loss)
        # count number of valid pixels (needed to get mean loss later):
        n_valid = tf.reduce_sum(valid_mask)
        # get BerHu loss:
        berhu_loss = tf.reduce_sum(loss_valid) / n_valid
    else:
        berhu_loss = tf.reduce_mean(loss)
    return berhu_loss


def l2_loss(label, prediction, lambda_, valid_mask=None):
    """
    Computes loss function (it compares ground truth (labels) to predictions
    y)
    :param label: ground truth label image
    :type label: np.array or tf tensor RGB image
    :param prediction: prediction (network output) of label image
    :type prediction: np.array or tf tensor RGB image
    :param lambda_: regularizer (least square loss if lambda_ = 0, scale
        invariant loss if lambda_ = 1, average of both if lambda_ = 0.5)
    :type lambda_: float (elemm [0, 1])
    :param valid_mask: binary map with 0 for invalid pixels that are not
        considered calculating the loss and 1 for valid pixels (default: None)
    :type valid_mask: np.array or tf tensor image
    :param log: PARAMETER DELETED calculate loss in log space (take elementwise
        natural log of label and prediction) (default: True)
    :type log: boolean

    """
#     if log:
#         # define offset to ensure that there will be no log(0)=-inf:
#         offset = 0.5
#         # natural logarithm (base e):
#         # take abs value to ensure that there will be no log(-x):
#         label = tf.log1p(tf.abs(label) + offset)
#         prediction = tf.log1p(tf.abs(prediction) + offset)

    diff = tf.abs(prediction - label)

    if valid_mask is not None:
        # get rid of invalid pixels (which do not contribute to loss):
        diff_valid = tf.multiply(valid_mask, diff)
        # count number of valid pixels (needed to get mean loss later):
        n_valid = tf.reduce_sum(valid_mask)
        mse = tf.reduce_sum(tf.square(diff_valid)) / n_valid
        reduced = tf.square(tf.reduce_sum(diff_valid) / n_valid)
    else:
        mse = tf.reduce_mean(tf.square(diff))
        reduced = tf.square(tf.reduce_mean(diff))

    return mse - lambda_ * reduced


def l1_loss(label, prediction, valid_mask=None):
    """
    Computes L1 loss function (it compares ground truth (labels) to predictions
    y)
    :param label: ground truth label image
    :type label: np.array or tf tensor RGB image
    :param prediction: prediction (network output) of label image
    :type prediction: np.array or tf tensor RGB image
    :param valid_mask: binary map with 0 for invalid pixels that are not
        considered calculating the loss and 1 for valid pixels (default: None)
    :type valid_mask: np.array or tf tensor image
    """
    diff = tf.abs(prediction - label)

    if valid_mask is not None:
        # get rid of invalid pixels (which do not contribute to loss):
        diff_valid = tf.multiply(valid_mask, diff)
        # count number of valid pixels (needed to get mean loss later):
        n_valid = tf.reduce_sum(valid_mask)
        loss = tf.reduce_sum(diff_valid) / n_valid
    else:
        loss = tf.reduce_mean(diff)

    return loss 


def loss_fct(label_albedo, label_shading, prediction_albedo,
             prediction_shading, lambda_, loss_type, valid_mask=None):
    """
    Computes loss function (it compares ground truth (labels) to predictions
    y)
    :param label_albedo: ground truth albedo label image
    :type label_albedo: np.array or tf tensor RGB image
    :param label_shading: ground truth shading label image
    :type label_shading: np.array or tf tensor RGB image
    :param prediction_albedo: prediction (network output) of albedo label image
    :type prediction_albedo: np.array or tf tensor RGB image
    :param prediction_shading: prediction (network output) of shading label
        image
    :type prediction_shading: np.array or tf tensor RGB image
    :param lambda_: regularizer (least square loss if lambda_ = 0, scale
        invariant loss if lambda_ = 1, average of both if lambda_ = 0.5)
    :type lambda_: float (elemm [0, 1])
    :param loss_type: elem {'l1', 'l2', 'berhu'}
    :type loss_type: str
    :param valid_mask: binary map with 0 for invalid pixels that are not
        considered calculating the loss and 1 for valid pixels (default: None)
    :type valid_mask: np.array or tf tensor image
    :param log: PARAMETER DELETED calculate loss in log space (take elementwise
        natural log of label and prediction) (default: True)
    :type log: boolean
    """
#     if log:
#        logstr = '_log'
#     else:
#         logstr = ''

    if loss_type == 'l2':
        loss_albedo = l2_loss(label=label_albedo, prediction=prediction_albedo,
                              lambda_=lambda_, valid_mask=valid_mask)
        loss_shading = l2_loss(label=label_shading,
                               prediction=prediction_shading, lambda_=lambda_,
                               valid_mask=valid_mask)
        if lambda_==0:
            lambda_str = loss_type  # + logstr
        elif lambda_==1:
            lambda_str = loss_type + '_invariant'  # + logstr
        elif lambda_==0.5:
            lambda_str = loss_type + '_avg'  # + logstr
    elif loss_type == 'l1':
        loss_albedo = l1_loss(label=label_albedo,
                              prediction=prediction_albedo,
                              valid_mask=valid_mask)
        loss_shading = l1_loss(label=label_shading,
                               prediction=prediction_shading,
                               valid_mask=valid_mask)
        lambda_str = loss_type

    elif loss_type == 'berhu':
        loss_albedo = berhu_loss(label=label_albedo,
                                 prediction=prediction_albedo,
                                 valid_mask=valid_mask)
        loss_shading = berhu_loss(label=label_shading,
                                  prediction=prediction_shading,
                                  valid_mask=valid_mask)
        lambda_str = loss_type  # + logstr
    else:
        raise TypeError("Enter valid loss_type ('l1', 'l2', 'berhu').")

    loss = loss_albedo + loss_shading
    # create summaries that give outputs (TensorFlow ops that output
    # protocol buffers containing 'summarized' data) of some scalar
    # parameters, like evolution of loss function (how does it change
    # over time? / rather not weights  because there are usually several
    # of them) etc (tf.summary.scalar())
    # (have a look at parameters that change over time (= training
    # steps))
    tf.summary.scalar(name='loss_', # + lambda_str,
                      tensor=loss)
    return loss


def compute_whdr_tf(reflectance, df_judgements, delta=0.1):
    """
    This is the Tensorflow implementation of the
    util.whdr_py3.compute_whdr() function
    Return the WHDR score for a reflectance image, evaluated against human
    judgements.  The return value is in the range 0.0 to 1.0.
    See section 3.5 of our paper for more details.
    :param reflectance: a tf tensor containing the linear RGB reflectance
        image.
    :type reflectance: tf. tensor of shape
        ['batch', 'height', 'width', channels]
    :param df_judgements: a pd.DataFrame object loaded and preprocessed from
        the Intrinsic Images in the Wild dataset.
    :param delta: the threshold where humans switch from saying "about the same"
        to "one point is darker."
    """
    # convert images in batch to greyscale (1 cheannel -> mean):
    imgs_grey_tf = tf.reduce_mean(reflectance, axis=3)
    # select pixels which are compared with each other (comp_p1_batch contains
    # all pixels that represent point1 and comp_p2_batch contains all pixels
    # that represent point2 over all samples in the batch. Points in
    # comp_p1_batch and comp_p2_batch are in the same order (important for
    # comparison)):
    comp_p1_batch = tf.gather_nd(imgs_grey_tf,
                                 [list(x) for x in df_judgements[['batch_nr',
                                                            'y1', 'x1']].values])
    comp_p2_batch = tf.gather_nd(imgs_grey_tf,
                                 [list(x) for x in df_judgements[['batch_nr',
                                                            'y2', 'x2']].values])
    # define/convert to pixel threshold:
    comp_p1_batch = tf.maximum(comp_p1_batch, 1e-10)
    comp_p2_batch = tf.maximum(comp_p2_batch, 1e-10)

    assert comp_p1_batch.shape==comp_p2_batch.shape, 'Missmatch in point ' + \
           'comparison length: ' + \
           'comp_p1_batch.shape={}, '.format(comp_p1_batch.shape) + \
           'comp_p2_batch.shape={}.'.format(comp_p2_batch.shape)
    # init tensors for human comparison score:
    # pixel compare to approx. same darkness:
    comparisons_eq = tf.zeros_like(comp_p1_batch, dtype=tf.int32)
    # pixel 1 is darker than px 2:
    comparisons_p1_darker = tf.ones_like(comp_p1_batch, dtype=tf.int32)
    # pixel 2 is darker than px 1:
    comparisons_p2_darker = tf.ones_like(comp_p1_batch, dtype=tf.int32) * 2
    # select 1 if px 1 is darker than px 2:
    px1_darker = tf.where(condition=comp_p2_batch / comp_p1_batch > 1.0 + delta,
                          x=comparisons_p1_darker, y=comparisons_eq)
    # select 2 if px 2 is darker than px 1:
    px2_darker = tf.where(condition=comp_p1_batch / comp_p2_batch > 1.0 + delta,
                          x=comparisons_p2_darker, y=comparisons_eq)
    # combine above by summing (-> 0 belongs to approx the same darkness):
    darker = px1_darker + px2_darker

    # get the human darker labels:
    human_labels = tf.constant(df_judgements['darker'].values, dtype=tf.int32)
    # compare human labels to algorithm darker predictions:
    hum_alg_comp = tf.not_equal(darker, human_labels)
    # list of weights/darker scores:
    darker_weights = tf.constant(df_judgements['darker_score'].values,
                                 dtype=tf.float32)
    # summed weights:
    weight_sum = tf.reduce_sum(darker_weights)
    # only select such weights that belong to prediction - human label mismatch:
    error_weights = tf.where(condition=hum_alg_comp,
                             x=darker_weights,
                             y=tf.zeros_like(hum_alg_comp, dtype=tf.float32))
    # sum all error_weights
    error_sum = tf.reduce_sum(error_weights)
    # get whdr over batch:
    return error_sum / weight_sum

