
# coding: utf-8

# In[1]:


"""
Import inference graph, extend it to a training graph and train network:

To extend the graph, we have to add:
    - input data structure
    - loss function
    - optimization op

- outdated: To plot all graphs directly in this notebook, run jupyter form
  terminal like this:
      jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000000
- continously tracking nvidia gpu: nvidia-smi -l 10
- converting a jupyter notebook file to python script: 
      jupyter nbconvert --to script cnns.ipynb
"""

import os   
import sys
sys.path.append('./util')
sys.path.append('./scripts_cnn_models_slim')
import time
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import slim
import matplotlib.pyplot as plt
import input_queues as iq
import cnn_model
import plot_helpers as plt_help
import general_helpers as ghelp
import nn_helpers as nnhelp
import cnn_helpers as chlp
import download
import resnet_v1 as resnet
import cnn_model as cnnm


__author__ = "Udo Dehm"
__copyright__ = "Copyright 2017"
__credits__ = ["Udo Dehm"]
__license__ = ""
__version__ = "0.1"
__maintainer__ = "Udo Dehm"
__email__ = "udo.dehm@mailbox.org"
__status__ = "Development"

__all__ = ['train_network_sintel', 'train_network_iiw', 'train_network']


# make only 'gpu:0' visible, so that only one gpu is used not both, see also
# https://github.com/tensorflow/tensorflow/issues/5066
# https://github.com/tensorflow/tensorflow/issues/3644#issuecomment-237631171
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def train_network_sintel(log_dir, data_dir, path_inference_graph, 
                         checkpoint_path, restore_scope, image_shape, 
                         initial_learning_rate, loss_opt, batch_size, 
                         num_epochs, display_step, save_step, nodes_name_dict,
                         logger, dataset='sintel', is_sample=False, norm=True, 
                         plot_inference_graph=False):
    """
    :param log_dir: path to directory for saving summary/log files
    :type log_dir: str
    :param data_dir: path to directory with training and validation data (in 
        this directory the training and validation csv files have to be
        located.)
    :type data_dir: str
    :param path_inference_graph: path to inference graph (graph without 
        'training' ops)
    :type path_inference_graph: str
    :param checkpoint_path: path to model parameters (checkpoint files
        e.g. 'logs/2/tfmodel-5' or None)
    :type checkpoint_path: str
    :param restore_scope: a scope name which has to be defined to find 
        parameters to restore (e.g. 'vgg_16'). Usually this is needed to
        find a pre-trained (tf slim) model within another model.
    :type restore_scope: str
    :param image_shape: Shape of images that should be used for training 
        (shape of cnn input tensor)
    :type image_shape: list with len(image_shape)==3
    :param initial_learning_rate: hyper-parameters for initial learning rate
    :type initial_learning_rate: float
    :param loss_opt: loss function used for optimization 
        ('berhu', 'l2', 'l2_inv', 'l2_avg')
    :type loss_opt: str
    :param batch_size: nr of data which is put through the network before 
        updating it, as default use: 16, 32 or 64. 
        batch_size determines how many data samples are loaded in the memory 
        (be careful with memory space)
    :type batch_size: int
    :param num_epochs: nr of times the training process loops through the 
        complete training data set (how often is the tr set 'seen')
        if you have 1000 training examples, and your batch size is 500, then it
        will take 2 iterations to complete 1 epoch.
    :type num_epochs: int
    :param display_step: every display_step'th training iteration information is
        printed to stdout and file training.log in log_dir (default: 100)
    :type display_step: int
    :param save_step: every save_step'th training iteration a summary file is 
        written to log_dir and checkpoint files are saved
    :type save_step: int
    :param nodes_name_dict: dictionary that contains name of input, albedo and 
        shading output in form {'input': '', 
                                'output_albedo': '',
                                'output_shading': ''}
    :type nodes_name_dict: dict
    :param dataset: which dataset to use for training (default: 'sintel')
    :type dataset: str, must be \elem {'iiw', 'sintel'}
    :param is_sample: flag, if True only a smaller sample size is used for 
        training and validation (default: False).
    :type is_sample: boolean
    :param norm: flag, if True image pixels are scaled to range [0, 1]
        (default: True)
    :type norm: boolean
    :param plot_inference_graph: flag, True if inference graph should be 
        plotted (default: False).
    :type plot_inference_graph: boolean
    """
    ############################################################################
    logger.info('Training on images of shape: {}'.format(image_shape))
    logger.info('Training on [0, 1] normalized pixel values: {}'.format(norm))
    logger.info('Initial learning rate: {}'.format(initial_learning_rate))
    logger.info('Loss function used for optimization: {}'.format(loss_opt))
    logger.info('Batch size: {}'.format(batch_size))
    logger.info('# epochs: {}'.format(num_epochs))
    logger.info('Write summary and checkpoints to file (in directory ' +
                '{}) every '.format(log_dir) +
                '{} iterations.'.format(save_step))

    # load meta graph (inference graph)
    # how to work with restored models:
    # https://www.tensorflow.org/programmers_guide/meta_graph
    # http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
    saver_restore = tf.train.import_meta_graph(path_inference_graph, 
                                               clear_devices=True)

    logger.debug('Restored inference graph from\n' +
                 '    {}'.format(path_inference_graph))
    variables_to_restore = slim.get_variables_to_restore(include=[restore_scope],
                                                         exclude=None)
    logger.info('# of parameters that can be restored: ' + 
                '{}.'.format(len(variables_to_restore)))

    ############################################################################

    # save default graph in variable:
    graph = tf.get_default_graph()
    if plot_inference_graph:
        # plot imported inference graph:
        plt_help.show_graph(graph.as_graph_def())

    # lets get the input
    x = graph.get_tensor_by_name(name=nodes_name_dict['input'])

    # setup target output classes (ground truth):
    y_albedo_label = tf.placeholder(dtype=tf.float32, 
                                    shape=[None] + image_shape, 
                                    name='out_albedo')
    y_shading_label = tf.placeholder(dtype=tf.float32, 
                                     shape=[None] + image_shape, 
                                     name='out_shading')

    # bool variable that indicates if we are in training mode (training=True) or
    # valid/test mode (training=False) this indicator is important if dropout 
    # or/and batch normalization is used.
    try:
        # try importing training node (is needed for models that use batch 
        # normalization etc.)
        training = graph.get_tensor_by_name(name='is_training_1:0')
        logger.debug('Was able to catch is_training node!')
    except KeyError:
        # elsewise just define a placeholder wich is used as dummy variable
        # and won't be used later:
        training = tf.placeholder(dtype=tf.bool, name='is_training_1')
    # define new global step variable:
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # get the last global training step if we continue training:
    try:
        last_global_step = int(checkpoint_path.split('/')[-1].split('-')[-1])
    except (ValueError, AttributeError):
        last_global_step = 0

    invalid_px_mask = tf.placeholder(dtype=tf.float32, 
                                     shape=[None] + image_shape, 
                                     name='invalid_px_mask')

    # get graph output nodes:
    y_albedo_pred = graph.get_tensor_by_name(name=nodes_name_dict['output_albedo'])
    y_shading_pred = graph.get_tensor_by_name(name=nodes_name_dict['output_shading'])
    # y_albedo_pred = tf.clip_by_value(t=y_albedo, clip_value_min=0, 
    #                                  clip_value_max=1, 
    #                                  name='0_1_clipping_albedo')
    # y_shading_pred = tf.clip_by_value(t=y_shading, clip_value_min=0,
    #                                   clip_value_max=1, 
    #                                   name='0_1_clipping_shading')

    ############################################################################
    ############################################################################

    with tf.name_scope('loss'):
        valid_mask = chlp.get_valid_pixels(image=x, 
                                           invalid_mask=invalid_px_mask)
        d = {'label_albedo': y_albedo_label,
             'label_shading': y_shading_label,
             'prediction_albedo': y_albedo_pred, 
             'prediction_shading': y_shading_pred,
             'valid_mask': valid_mask}

        loss_dict = {'berhu': chlp.sintel_loss_fct(**d, **{'loss_type': 'berhu', 
                                                           'lambda_': None}),
                     'l1': chlp.sintel_loss_fct(**d, **{'loss_type': 'l1',
                                                        'lambda_': None}),
                     'l2': chlp.sintel_loss_fct(**d, **{'loss_type': 'l2', 
                                                        'lambda_': 0}),
                     'l2_inv': chlp.sintel_loss_fct(**d, **{'loss_type': 'l2',
                                                            'lambda_': 1}),
                     'l2_avg': chlp.sintel_loss_fct(**d, **{'loss_type': 'l2',
                                                            'lambda_': 0.5})
                    }
        if loss_opt not in ('berhu', 'l1', 'l2', 'l2_inv', 'l2_avg'):
            raise ValueError('{} is not a valid loss '.format(loss_opt) + 
                             'function. Set parameter loss_opt to one of the ' +
                             "following: ('berhu', 'l1', 'l2', 'l2_inv', " +
                             "'l2_avg')")

        loss = loss_dict[loss_opt]
    logger.debug('Defined training losses.')

    ############################################################################

    # Use an AdamOptimizer to train the network:
    with tf.name_scope('optimization'):
        optimizer = tf.train.AdamOptimizer(learning_rate=initial_learning_rate)
        
        # use slim optimizaton op only if a batch normalization is in the 
        # tf graph.
        use_slim_train_opt = False
        for v in graph.as_graph_def().node:
            if 'batchnorm' in v.name.lower():
                use_slim_train_opt = True
                break
        # Many networks utilize modules, like BatchNorm, that require 
        # performing a series of non-gradient updates during training. 
        # slim.learning.create_train_op allows a user to pass in a list of 
        # update_ops to call along with the gradient updates.
        #   train_op = slim.learning.create_train_op(total_loss, optimizer,
        #                                            update_ops)
        # By default, slim.learning.create_train_op includes all update ops 
        # that are part of the `tf.GraphKeys.UPDATE_OPS` collection. 
        # Additionally, TF-Slim's slim.batch_norm function adds the moving mean
        # and moving variance updates to this collection. Consequently, users 
        # who want to use slim.batch_norm will not need to take any additional
        # steps in order to have the moving mean and moving variance updates be
        # computed.
        # (see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/learning.py)
        if use_slim_train_opt:
            opt_step = slim.learning.create_train_op(total_loss=loss, 
                                                     optimizer=optimizer,
                                                     global_step=global_step) 
            logger.debug('Using slim optimizaton op because a batch ' +
                         'normalization is apparently in the tf graph.')
        else:
            opt_step = optimizer.minimize(loss, global_step=global_step)
            logger.debug('Using default tf optimizaton op because there is ' +
                         'apparently no batch normalization in the tf ' +
                         'graph.')

    logger.debug('Defined optimization method.')
    
    
    ############################################################################

    # to get every summary defined above we merge them to get one target:
    merge_train_summaries = tf.summary.merge_all()
    # define a FileWriter op which writes summaries defined above to disk:
    summary_writer = tf.summary.FileWriter(log_dir)
    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(max_to_keep=None)

    ############################################################################

    # introduce some validation set specific summaries
    # These summaries need to be defined blow the function 
    # merge_train_summaries = tf.summary.merge_all()
    # because the validation set summaries are added to the summary writer at 
    # different times. If they had been summarized with the training summaries 
    # they would have to be defined at times where merge_train_summaries are 
    # added to the summary writer
    with tf.name_scope('loss/valid/'):
        valid_dict = {key: tf.placeholder(dtype=tf.float32, name=key) for                       key in loss_dict.keys()}
        valid_summaries = [tf.summary.scalar(name=key, tensor=val) for                            key, val in valid_dict.items()]
        valid_sums_merged = tf.summary.merge(inputs=valid_summaries,
                                             collections=None, name=None)
    logger.debug('Defined validation losses.')
    logger.debug('Finished building training graph.')
    logger.info('Total parameters of network: ' +
                '{}'.format(nnhelp.network_params()))

    ############################################################################
    ############################################################################

    # import data:
    if is_sample:
        sample = 'sample_'
    else:
        sample = ''
    
    # import training data:
    file = sample + 'data_sintel_shading_train.csv'
    df_train = pd.read_csv(filepath_or_buffer=data_dir + file, 
                           sep=',', header=None,
                           names=['img', 'alb', 'shad', 'invalid'])
    # complete image paths:
    df_train = data_dir + df_train

    # # enable this line to train on only one image:
    # df_train1 = df_train.loc[[0]]
    # # replicate this row 100 times:
    # df_train = pd.concat([df_train1]*100).reset_index(drop=True)

    # instantiate a data queue for feeding data in (mini) batches to cnn:
    data_train = iq.DataQueue(df=df_train, batch_size=batch_size,
                              num_epochs=num_epochs)

    logger.debug('Imported {} training data '.format(dataset) + 
                 '(#: {}) '.format(data_train.df.shape[0]) + 
                 'from\n    {}'.format(data_dir + file))
    
    # import validation data set: 
    # why not using the whole validation set for validation at once? 
    # - limited memory space.
    #  -> After each training epoch we will use the complete validation dataset
    #     to calculate the error/accuracy on the validation set
    file = sample + 'data_sintel_shading_valid.csv'
    df_valid = pd.read_csv(filepath_or_buffer=data_dir + file, 
                           sep=',', header=None,
                           names=['img', 'alb', 'shad', 'invalid'])
    # complete image paths:
    df_valid = data_dir + df_valid
    # instantiate a data queue for feeding data in (mini) batches to cnn:
    data_valid = iq.DataQueue(df=df_valid, batch_size=batch_size,
                              num_epochs=num_epochs)

    logger.debug('Imported {} validation data '.format(dataset) +
                 '(#: {}) '.format(data_valid.df.shape[0]) + 
                 'from\n    {}'.format(data_dir + file))

    ############################################################################
    ############################################################################

    logger.info('Start training:')
    # Initialization:
    # Op that initializes global variables in the graph:
    init_global = tf.global_variables_initializer()
    # Op that initializes local variables in the graph:
    init_local = tf.local_variables_initializer()

    config = tf.ConfigProto(device_count = {'GPU': 1},
                            intra_op_parallelism_threads=4
    #                        allow_soft_placement = True,
    #                        log_device_placement=False
                           )
    with tf.Session(config=config) as sess: 
        
        ########################################################################
        
        # initialize all variables:
        sess.run([init_global, init_local])

        # assign the last true global step to to global step:
        sess.run(global_step.assign(last_global_step))
        logger.info('Assigned last global training step: ' +
                    '{}'.format(global_step.eval()))

        if checkpoint_path:
            try:
                # restore saved model parameters (weights, biases, etc):
                saver_restore.restore(sess, checkpoint_path)
                logger.info('Restoring parameters from ' +
                            '{}'.format(checkpoint_path))
            except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError):
                # in the worst case parameters are loaded twice.
                # restore the parameters:
                init_fn = slim.assign_from_checkpoint_fn(checkpoint_path,
                                                         variables_to_restore)
                init_fn(sess)
                logger.info('Restoring parameters from ' +
                            '{}'.format(checkpoint_path))

        # Adds a Graph to the event file.
        # create summary that give output (TensorFlow op that output protocol 
        # buffers containing 'summarized' data) of the built Tensorflow graph:
        summary_writer.add_graph(sess.graph)

        # start timer for total training time:
        start_total_time = time.time()
        # set timer to measure the displayed training steps:
        start_time = start_total_time

        ########################################################################
        
        # Training:
        # train loop
        # train until all data is processed (queue empty),
        # number of iterations depends on number of data, number of epochs and 
        # batch size:
        iter_start = data_train.iter_left
        logger.info('For training it takes {}\n'.format(iter_start) +
                    '    (= # data / batch_size * epochs) iterations to loop ' +
                    'through {} samples of\n  '.format(data_train.df.shape[0]) +
                    '  training data over {} '.format(data_train.num_epochs) +
                    'epochs summarized in batches of size ' + 
                    '{}.\n'.format(data_train.batch_size) +
                    '    So, there are # data / batch_size = ' +
                    '{}'.format(int(data_train.df.shape[0]/data_train.batch_size))+
                    ' iterations per epoch.')

        while data_train.iter_left >= 0:
            try:
                # take a (mini) batch of the training data:
                deq_train = data_train.dequeue()
                img_b, alb_b, shad_b, inv_b = iq.next_batch_sintel(deq=deq_train,
                                                                   output_shape=image_shape,
                                                                   is_scale=True,
                                                                   is_flip=True,
                                                                   is_rotated=True,
                                                                   norm=norm)
                # run training/optimization step:
                # Run one step of the model.  The return values are the 
                # activations from the `train_op` (which is discarded) and the 
                # `loss` Op.  To inspect the values of your Ops or variables, 
                # you may include them in the list passed to sess.run() and the 
                # value tensors will be returned in the tuple from the call.
                feed_dict_tr = {x: img_b,
                                y_albedo_label: alb_b,
                                y_shading_label: shad_b,
                                invalid_px_mask: inv_b,
                                training: True}
                sess.run(opt_step, feed_dict=feed_dict_tr)

                ################################################################

                # report training set accuracy every display_step-th step:
                if (data_train.num_iter) % display_step == 0:
                    # console output:
                    train_loss_dict = {}
                    for key, val in loss_dict.items():
                        train_loss_dict[key] = sess.run(val, 
                                                        feed_dict=feed_dict_tr)

                    dur_time = time.time() - start_time
                    dur_time = ghelp.get_time_format(time_in_sec=dur_time)
                    dur_time = ghelp.time_tuple_to_str(time_tuple=dur_time)
                    logger.info('step {}: '.format(data_train.num_iter) +
                                'training ({}) loss '.format(loss_opt) +
                                '{:.4f}'.format(train_loss_dict[loss_opt]) + 
                                '\n    (' +
                                ', '.join(['{}: {:.4f}'.format(it[0], it[1]) \
                                           for it in train_loss_dict.items() \
                                           if it[0]!=loss_opt]) +
                                ', ET: {}).'.format(dur_time))
                    # reset timer to measure the displayed training steps:
                    start_time = time.time()

                ################################################################

                # Validation:
                # display validation set accuracy and loss after each completed 
                # epoch (1 epoch ^= data_train.df.shape[0]/data_train.batch_size
                # training steps => steps per epoch)
                val_epoch = int(data_train.df.shape[0] / data_train.batch_size)
                
                if data_train.num_iter % val_epoch == 0:
                    # After each training epoch we will use the complete 
                    # validation data set to calculate the error/accuracy on the
                    # validation set:
                    # loop through one validation data set epoch:
                    # initialize dictionary which contains all validation 
                    # losses:
                    valid_loss_dict = dict.fromkeys(loss_dict, 0)
                    valid_steps_per_epoch = int(data_valid.df.shape[0] / 
                                                data_valid.batch_size)
                    for j in range(valid_steps_per_epoch):
                        # DISCLAIMER: we do not run the opt_step here (on 
                        # the validation data set) because we do not want to 
                        # train our network on the validation set. Important 
                        # for batch normalization and dropout 
                        # (training -> False).
                        # get validation data set (mini) batch:
                        lst = iq.next_batch_sintel(deq=data_valid.dequeue(), 
                                                   output_shape=image_shape,
                                                   is_scale=False,
                                                   is_flip=False,
                                                   is_rotated=False,
                                                   norm=norm)
                        img_b_val, alb_b_val, shad_b_val, inv_b_val = lst

                        # calculate the mean loss of this validation batch and 
                        # sum it with the previous mean batch losses:
                        fd_val = {x: img_b_val,
                                  y_albedo_label: alb_b_val,
                                  y_shading_label: shad_b_val,
                                  invalid_px_mask: inv_b_val,
                                  training: False}

                        for key, val in loss_dict.items():
                            # divide each loss loss by the iteration steps 
                            # (steps_per_epoch) to get the mean val loss:
                            mean_val = val / valid_steps_per_epoch
                            valid_loss_dict[key] += sess.run(mean_val, 
                                                             feed_dict=fd_val)
                    # adding a mean loss summary op (for tensorboard):
                    feed_dict_vl = {valid_dict[key]: valid_loss_dict[key] for                                     key in valid_dict.keys()}
                    val_loss_sums = sess.run(valid_sums_merged,
                                             feed_dict=feed_dict_vl)
                    summary_writer.add_summary(summary=val_loss_sums, 
                                               global_step=global_step.eval())

                    dur_time = time.time() - start_time
                    dur_time = ghelp.get_time_format(time_in_sec=dur_time)
                    dur_time = ghelp.time_tuple_to_str(time_tuple=dur_time)
                    logger.info('step {} '.format(data_train.num_iter) +
                                '(epoch ' + 
                                '{}):'.format(data_train.completed_epochs + 1) +
                                ' mean validation losses:\n    ' +
                                ', '.join(['{}: {:.4f}'.format(it[0], it[1]) \
                                           for it in valid_loss_dict.items()]) +
                                ' (ET: {}).'.format(dur_time))
                    # reset timer to measure the displayed training steps:
                    start_time = time.time()

                ################################################################

                if data_train.num_iter % save_step == 0:
                    # save checkpoint files to disk:
                    save_path = saver.save(sess, log_dir + 'tfmodel',
                                           global_step=global_step.eval())
                    s = sess.run(merge_train_summaries, feed_dict=feed_dict_tr)
                    # adds a Summary protocol buffer to the event file 
                    # (global_step: Number. Optional global step value to record
                    # with the summary. Each stepp i is assigned to the 
                    # corresponding summary parameter.)
                    summary_writer.add_summary(summary=s, 
                                               global_step=global_step.eval())
                    dur_time = time.time() - start_time
                    dur_time = ghelp.get_time_format(time_in_sec=dur_time)
                    dur_time = ghelp.time_tuple_to_str(time_tuple=dur_time)
                    logger.info('Saved data (step ' + 
                                '{}):\n'.format(data_train.num_iter) +
                                '    Checkpoint file written to: ' + 
                                '{} '.format(save_path) +
                                '(ET: {}).'.format(dur_time))
                    # reset timer to measure the displayed training steps:
                    start_time = time.time()

            # end while loop when there are no elements left to dequeue:
            except IndexError:
                end_total_time = time.time() - start_total_time
                end_total_time = ghelp.get_time_format(end_total_time)
                end_total_time = ghelp.time_tuple_to_str(time_tuple=end_total_time)
                logger.info('Training done... total training time: ' + 
                            '{}.'.format(end_total_time))
                break

    logger.info('Finished training.')


# In[2]:


# def train_network_sintel_upproject(log_dir, data_dir, path_inference_graph, 
#                          checkpoint_path, restore_scope, image_shape, 
#                          initial_learning_rate, loss_opt, batch_size, 
#                          num_epochs, display_step, save_step, nodes_name_dict,
#                          logger, dataset='sintel', is_sample=False, norm=True, 
#                          plot_inference_graph=False):
#     """
#     :param log_dir: path to directory for saving summary/log files
#     :type log_dir: str
#     :param data_dir: path to directory with training and validation data (in 
#         this directory the training and validation csv files have to be
#         located.)
#     :type data_dir: str
#     :param path_inference_graph: path to inference graph (graph without 
#         'training' ops)
#     :type path_inference_graph: str
#     :param checkpoint_path: path to model parameters (checkpoint files
#         e.g. 'logs/2/tfmodel-5' or None)
#     :type checkpoint_path: str
#     :param restore_scope: a scope name which has to be defined to find 
#         parameters to restore (e.g. 'vgg_16'). Usually this is needed to
#         find a pre-trained (tf slim) model within another model.
#     :type restore_scope: str
#     :param image_shape: Shape of images that should be used for training 
#         (shape of cnn input tensor)
#     :type image_shape: list with len(image_shape)==3
#     :param initial_learning_rate: hyper-parameters for initial learning rate
#     :type initial_learning_rate: float
#     :param loss_opt: loss function used for optimization 
#         ('berhu', 'l2', 'l2_inv', 'l2_avg')
#     :type loss_opt: str
#     :param batch_size: nr of data which is put through the network before 
#         updating it, as default use: 16, 32 or 64. 
#         batch_size determines how many data samples are loaded in the memory 
#         (be careful with memory space)
#     :type batch_size: int
#     :param num_epochs: nr of times the training process loops through the 
#         complete training data set (how often is the tr set 'seen')
#         if you have 1000 training examples, and your batch size is 500, then it
#         will take 2 iterations to complete 1 epoch.
#     :type num_epochs: int
#     :param display_step: every display_step'th training iteration information is
#         printed to stdout and file training.log in log_dir (default: 100)
#     :type display_step: int
#     :param save_step: every save_step'th training iteration a summary file is 
#         written to log_dir and checkpoint files are saved
#     :type save_step: int
#     :param nodes_name_dict: dictionary that contains name of input, albedo and 
#         shading output in form {'input': '', 
#                                 'output_albedo': '',
#                                 'output_shading': ''}
#     :type nodes_name_dict: dict
#     :param dataset: which dataset to use for training (default: 'sintel')
#     :type dataset: str, must be \elem {'iiw', 'sintel'}
#     :param is_sample: flag, if True only a smaller sample size is used for 
#         training and validation (default: False).
#     :type is_sample: boolean
#     :param norm: flag, if True image pixels are scaled to range [0, 1]
#         (default: True)
#     :type norm: boolean
#     :param plot_inference_graph: flag, True if inference graph should be 
#         plotted (default: False).
#     :type plot_inference_graph: boolean
#     """
#     ############################################################################
#     logger.info('Training on images of shape: {}'.format(image_shape))
#     logger.info('Training on [0, 1] normalized pixel values: {}'.format(norm))
#     logger.info('Initial learning rate: {}'.format(initial_learning_rate))
#     logger.info('Loss function used for optimization: {}'.format(loss_opt))
#     logger.info('Batch size: {}'.format(batch_size))
#     logger.info('# epochs: {}'.format(num_epochs))
#     logger.info('Write summary and checkpoints to file (in directory ' +
#                 '{}) every '.format(log_dir) +
#                 '{} iterations.'.format(save_step))

    
#     tf.reset_default_graph()

#     ############################################################################



#     # input for cnn:
#     x = tf.placeholder(dtype=tf.float32, 
#                        shape=[batch_size] + image_shape, 
#                        name='input')    
#     training = tf.placeholder(dtype=tf.bool, name='is_training')    
    
    
#     bn_decay = 0.999
#     bn_epsilon = 0.001
#     with tf.Graph().as_default():    
#         # Create the model, use the default arg scope to configure the batch norm parameters.
#         with slim.arg_scope(resnet.resnet_arg_scope()):
#             # 1000 classes instead of 1001.
#             scope = 'resnet_v1_50'
#             net, end_points = resnet.resnet_v1_50(inputs=x,
#                                                   num_classes=None,
#                                                   is_training=training,
#                                                   global_pool=False,
#                                                   output_stride=None,
#                                                   spatial_squeeze=False,
#                                                   reuse=None,
#                                                   scope=scope)
#             print(net)

#         with tf.variable_scope(name_or_scope='decoder', default_name='decoder',
#                                values=[net]) as sc:
#             end_points_collection = sc.name + '_end_points'
#             # Collect outputs for conv2d, fully_connected and max_pool2d.
#             with slim.arg_scope(cnnm.upproject_arg_scope(weight_decay=0.0001,
#                                                          batch_norm_decay=bn_decay,
#                                                          batch_norm_epsilon=bn_epsilon,
#                                                          batch_norm_scale=True,
#                                                          activation_fn=None,
#                                                          use_batch_norm=True,
#                                                          is_training=training)):
#                 with slim.arg_scope([slim.conv2d],
#                                     outputs_collections=end_points_collection):
#                     net = slim.conv2d(inputs=net,
#                                       num_outputs=1024,
#                                       kernel_size=[1, 1],
#                                       stride=1,
#                                       padding='SAME',
#                                       data_format='NHWC', 
#                                       trainable=True,
#                                       scope='conv1')

#             net = cnnm.up_project(input_data=net, kernel_size=[3, 3], num_outputs=512, 
#                                   batch_size=batch_size, scope='2x_up_projection', stride=1,
#                                   use_batch_norm=True, bn_decay=bn_decay,
#                                   bn_epsilon=bn_epsilon, is_training=training)
#             print(net)
#             net = cnnm.up_project(input_data=net, kernel_size=[3, 3], num_outputs=256, 
#                                   batch_size=batch_size, scope='4x_up_projection', stride=1,
#                                   use_batch_norm=True, bn_decay=bn_decay,
#                                   bn_epsilon=bn_epsilon, is_training=training)
#             net = cnnm.up_project(input_data=net, kernel_size=[3, 3], num_outputs=128, 
#                                   batch_size=batch_size, scope='8x_up_projection', stride=1,
#                                   use_batch_norm=True, bn_decay=bn_decay,
#                                   bn_epsilon=bn_epsilon, is_training=training)
#             net = cnnm.up_project(input_data=net, kernel_size=[3, 3], num_outputs=64, 
#                                   batch_size=batch_size, scope='16x_up_projection', stride=1, 
#                                   use_batch_norm=True, bn_decay=bn_decay,
#                                   bn_epsilon=bn_epsilon, is_training=training)

#             net_albedo = cnnm.up_project(input_data=net, kernel_size=[3, 3], num_outputs=32, 
#                                          batch_size=batch_size, scope='32x_up_projection_albedo', stride=1, 
#                                          use_batch_norm=True, bn_decay=bn_decay,
#                                          bn_epsilon=bn_epsilon, is_training=training)
#             net_shading = cnnm.up_project(input_data=net, kernel_size=[3, 3], num_outputs=32, 
#                                           batch_size=batch_size, scope='32x_up_projection_shading', stride=1, 
#                                           use_batch_norm=True, bn_decay=bn_decay,
#                                           bn_epsilon=bn_epsilon, is_training=training)

#             with slim.arg_scope(cnnm.upproject_arg_scope(weight_decay=0.0001,
#                                                          batch_norm_decay=bn_decay,
#                                                          batch_norm_epsilon=bn_epsilon,
#                                                          batch_norm_scale=True,
#                                                          activation_fn=None,
#                                                          use_batch_norm=True,
#                                                          is_training=training)):
#                 with slim.arg_scope([slim.conv2d],
#                                     outputs_collections=end_points_collection):
#                     y_albedo_pred = slim.conv2d(inputs=net_albedo,
#                                              num_outputs=3,
#                                              kernel_size=[3, 3],
#                                              stride=1,
#                                              padding='SAME',
#                                              data_format='NHWC', 
#                                              trainable=True,
#                                              scope='output_albedo_prediction')

#                     y_shading_pred = slim.conv2d(inputs=net_shading,
#                                               num_outputs=3,
#                                               kernel_size=[3, 3],
#                                               stride=1,
#                                               padding='SAME',
#                                               data_format='NHWC', 
#                                               trainable=True,
#                                               scope='output_shading_prediction')

#     # setup target output classes (ground truth):
#     y_albedo_label = tf.placeholder(dtype=tf.float32, 
#                                     shape=[None] + image_shape, 
#                                     name='out_albedo')
#     y_shading_label = tf.placeholder(dtype=tf.float32, 
#                                      shape=[None] + image_shape, 
#                                      name='out_shading')
    
#     variables_to_restore = slim.get_variables_to_restore(include=[restore_scope],
#                                                          exclude=None)
#     logger.info('# of parameters that can be restored: ' + 
#                 '{}.'.format(len(variables_to_restore)))

    
    
    


#     # save default graph in variable:
#     graph = tf.get_default_graph()
#     if plot_inference_graph:
#         # plot imported inference graph:
#         plt_help.show_graph(graph.as_graph_def())
    
#     # define new global step variable:
#     global_step = tf.Variable(0, name='global_step', trainable=False)
#     # get the last global training step if we continue training:
#     try:
#         last_global_step = int(checkpoint_path.split('/')[-1].split('-')[-1])
#     except (ValueError, AttributeError):
#         last_global_step = 0

#     invalid_px_mask = tf.placeholder(dtype=tf.float32, 
#                                      shape=[None] + image_shape, 
#                                      name='invalid_px_mask')

#     ############################################################################
#     ############################################################################

#     with tf.name_scope('loss'):
#         valid_mask = chlp.get_valid_pixels(image=x, 
#                                            invalid_mask=invalid_px_mask)
#         d = {'label_albedo': y_albedo_label,
#              'label_shading': y_shading_label,
#              'prediction_albedo': y_albedo_pred, 
#              'prediction_shading': y_shading_pred,
#              'valid_mask': valid_mask}

#         loss_dict = {'berhu': chlp.sintel_loss_fct(**d, **{'loss_type': 'berhu', 
#                                                            'lambda_': None}),
#                      'l1': chlp.sintel_loss_fct(**d, **{'loss_type': 'l1',
#                                                         'lambda_': None}),
#                      'l2': chlp.sintel_loss_fct(**d, **{'loss_type': 'l2', 
#                                                         'lambda_': 0}),
#                      'l2_inv': chlp.sintel_loss_fct(**d, **{'loss_type': 'l2',
#                                                             'lambda_': 1}),
#                      'l2_avg': chlp.sintel_loss_fct(**d, **{'loss_type': 'l2',
#                                                             'lambda_': 0.5})
#                     }
#         if loss_opt not in ('berhu', 'l1', 'l2', 'l2_inv', 'l2_avg'):
#             raise ValueError('{} is not a valid loss '.format(loss_opt) + 
#                              'function. Set parameter loss_opt to one of the ' +
#                              "following: ('berhu', 'l1', 'l2', 'l2_inv', " +
#                              "'l2_avg')")

#         loss = loss_dict[loss_opt]
#     logger.debug('Defined training losses.')

#     ############################################################################

#     # Use an AdamOptimizer to train the network:
#     with tf.name_scope('optimization'):
#         optimizer = tf.train.AdamOptimizer(learning_rate=initial_learning_rate)
        
#         # use slim optimizaton op only if a batch normalization is in the 
#         # tf graph.
#         use_slim_train_opt = False
#         for v in graph.as_graph_def().node:
#             if 'batchnorm' in v.name.lower():
#                 use_slim_train_opt = True
#                 break
#         # Many networks utilize modules, like BatchNorm, that require 
#         # performing a series of non-gradient updates during training. 
#         # slim.learning.create_train_op allows a user to pass in a list of 
#         # update_ops to call along with the gradient updates.
#         #   train_op = slim.learning.create_train_op(total_loss, optimizer,
#         #                                            update_ops)
#         # By default, slim.learning.create_train_op includes all update ops 
#         # that are part of the `tf.GraphKeys.UPDATE_OPS` collection. 
#         # Additionally, TF-Slim's slim.batch_norm function adds the moving mean
#         # and moving variance updates to this collection. Consequently, users 
#         # who want to use slim.batch_norm will not need to take any additional
#         # steps in order to have the moving mean and moving variance updates be
#         # computed.
#         # (see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/learning.py)
#         if use_slim_train_opt:
#             opt_step = slim.learning.create_train_op(total_loss=loss, 
#                                                      optimizer=optimizer,
#                                                      global_step=global_step) 
#             logger.debug('Using slim optimizaton op because a batch ' +
#                          'normalization is apparently in the tf graph.')
#         else:
#             opt_step = optimizer.minimize(loss, global_step=global_step)
#             logger.debug('Using default tf optimizaton op because there is ' +
#                          'apparently no batch normalization in the tf ' +
#                          'graph.')

#     logger.debug('Defined optimization method.')
    
    
#     ############################################################################

#     # to get every summary defined above we merge them to get one target:
#     merge_train_summaries = tf.summary.merge_all()
#     # define a FileWriter op which writes summaries defined above to disk:
#     summary_writer = tf.summary.FileWriter(log_dir)
#     # Create a saver for writing training checkpoints.
#     saver = tf.train.Saver(max_to_keep=None)

#     ############################################################################

#     # introduce some validation set specific summaries
#     # These summaries need to be defined blow the function 
#     # merge_train_summaries = tf.summary.merge_all()
#     # because the validation set summaries are added to the summary writer at 
#     # different times. If they had been summarized with the training summaries 
#     # they would have to be defined at times where merge_train_summaries are 
#     # added to the summary writer
#     with tf.name_scope('loss/valid/'):
#         valid_dict = {key: tf.placeholder(dtype=tf.float32, name=key) for \
#                       key in loss_dict.keys()}
#         valid_summaries = [tf.summary.scalar(name=key, tensor=val) for \
#                            key, val in valid_dict.items()]
#         valid_sums_merged = tf.summary.merge(inputs=valid_summaries,
#                                              collections=None, name=None)
#     logger.debug('Defined validation losses.')
#     logger.debug('Finished building training graph.')
#     logger.info('Total parameters of network: ' +
#                 '{}'.format(nnhelp.network_params()))

#     ############################################################################
#     ############################################################################

#     # import data:
#     if is_sample:
#         sample = 'sample_'
#     else:
#         sample = ''
    
#     # import training data:
#     file = sample + 'data_sintel_shading_train.csv'
#     df_train = pd.read_csv(filepath_or_buffer=data_dir + file, 
#                            sep=',', header=None,
#                            names=['img', 'alb', 'shad', 'invalid'])
#     # complete image paths:
#     df_train = data_dir + df_train

#     # # enable this line to train on only one image:
#     # df_train1 = df_train.loc[[0]]
#     # # replicate this row 100 times:
#     # df_train = pd.concat([df_train1]*100).reset_index(drop=True)

#     # instantiate a data queue for feeding data in (mini) batches to cnn:
#     data_train = iq.DataQueue(df=df_train, batch_size=batch_size,
#                               num_epochs=num_epochs)

#     logger.debug('Imported {} training data '.format(dataset) + 
#                  '(#: {}) '.format(data_train.df.shape[0]) + 
#                  'from\n    {}'.format(data_dir + file))
    
#     # import validation data set: 
#     # why not using the whole validation set for validation at once? 
#     # - limited memory space.
#     #  -> After each training epoch we will use the complete validation dataset
#     #     to calculate the error/accuracy on the validation set
#     file = sample + 'data_sintel_shading_valid.csv'
#     df_valid = pd.read_csv(filepath_or_buffer=data_dir + file, 
#                            sep=',', header=None,
#                            names=['img', 'alb', 'shad', 'invalid'])
#     # complete image paths:
#     df_valid = data_dir + df_valid
#     # instantiate a data queue for feeding data in (mini) batches to cnn:
#     data_valid = iq.DataQueue(df=df_valid, batch_size=batch_size,
#                               num_epochs=num_epochs)

#     logger.debug('Imported {} validation data '.format(dataset) +
#                  '(#: {}) '.format(data_valid.df.shape[0]) + 
#                  'from\n    {}'.format(data_dir + file))

#     ############################################################################
#     ############################################################################

#     logger.info('Start training:')
#     # Initialization:
#     # Op that initializes global variables in the graph:
#     init_global = tf.global_variables_initializer()
#     # Op that initializes local variables in the graph:
#     init_local = tf.local_variables_initializer()

#     config = tf.ConfigProto(device_count = {'GPU': 1},
#                             intra_op_parallelism_threads=4
#     #                        allow_soft_placement = True,
#     #                        log_device_placement=False
#                            )
#     with tf.Session(config=config) as sess: 
        
#         ########################################################################
        
#         # initialize all variables:
#         sess.run([init_global, init_local])

#         # assign the last true global step to to global step:
#         sess.run(global_step.assign(last_global_step))
#         logger.info('Assigned last global training step: ' +
#                     '{}'.format(global_step.eval()))

#         if checkpoint_path:
#             try:
#                 # restore saved model parameters (weights, biases, etc):
#                 saver_restore.restore(sess, checkpoint_path)
#                 logger.info('Restoring parameters from ' +
#                             '{}'.format(checkpoint_path))
#             except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError):
#                 # in the worst case parameters are loaded twice.
#                 # restore the parameters:
#                 init_fn = slim.assign_from_checkpoint_fn(checkpoint_path,
#                                                          variables_to_restore)
#                 init_fn(sess)
#                 logger.info('Restoring parameters from ' +
#                             '{}'.format(checkpoint_path))

#         # Adds a Graph to the event file.
#         # create summary that give output (TensorFlow op that output protocol 
#         # buffers containing 'summarized' data) of the built Tensorflow graph:
#         summary_writer.add_graph(sess.graph)

#         # start timer for total training time:
#         start_total_time = time.time()
#         # set timer to measure the displayed training steps:
#         start_time = start_total_time

#         ########################################################################
        
#         # Training:
#         # train loop
#         # train until all data is processed (queue empty),
#         # number of iterations depends on number of data, number of epochs and 
#         # batch size:
#         iter_start = data_train.iter_left
#         logger.info('For training it takes {}\n'.format(iter_start) +
#                     '    (= # data / batch_size * epochs) iterations to loop ' +
#                     'through {} samples of\n  '.format(data_train.df.shape[0]) +
#                     '  training data over {} '.format(data_train.num_epochs) +
#                     'epochs summarized in batches of size ' + 
#                     '{}.\n'.format(data_train.batch_size) +
#                     '    So, there are # data / batch_size = ' +
#                     '{}'.format(int(data_train.df.shape[0]/data_train.batch_size))+
#                     ' iterations per epoch.')

#         while data_train.iter_left >= 0:
#             try:
#                 # take a (mini) batch of the training data:
#                 deq_train = data_train.dequeue()
#                 if dataset=='sintel':
#                     img_b, alb_b, shad_b, inv_b = iq.next_batch_sintel(deq=deq_train,
#                                                                        output_shape=image_shape,
#                                                                        is_scale=True,
#                                                                        is_flip=True,
#                                                                        is_rotated=True,
#                                                                        norm=norm)
#                     # run training/optimization step:
#                     # Run one step of the model.  The return values are the 
#                     # activations from the `train_op` (which is discarded) and the 
#                     # `loss` Op.  To inspect the values of your Ops or variables, 
#                     # you may include them in the list passed to sess.run() and the 
#                     # value tensors will be returned in the tuple from the call.
#                     feed_dict_tr = {x: img_b,
#                                     y_albedo_label: alb_b,
#                                     y_shading_label: shad_b,
#                                     invalid_px_mask: inv_b,
#                                     training: True}
                    
#                 sess.run(opt_step, feed_dict=feed_dict_tr)

#                 ################################################################

#                 # report training set accuracy every display_step-th step:
#                 if (data_train.num_iter) % display_step == 0:
#                     # console output:
#                     train_loss_dict = {}
#                     for key, val in loss_dict.items():
#                         train_loss_dict[key] = sess.run(val, 
#                                                         feed_dict=feed_dict_tr)

#                     dur_time = time.time() - start_time
#                     dur_time = ghelp.get_time_format(time_in_sec=dur_time)
#                     dur_time = ghelp.time_tuple_to_str(time_tuple=dur_time)
#                     logger.info('step {}: '.format(data_train.num_iter) +
#                                 'training ({}) loss '.format(loss_opt) +
#                                 '{:.4f}'.format(train_loss_dict[loss_opt]) + 
#                                 '\n    (' +
#                                 ', '.join(['{}: {:.4f}'.format(it[0], it[1]) \
#                                            for it in train_loss_dict.items() \
#                                            if it[0]!=loss_opt]) +
#                                 ', ET: {}).'.format(dur_time))
#                     # reset timer to measure the displayed training steps:
#                     start_time = time.time()

#                 ################################################################

#                 # Validation:
#                 # display validation set accuracy and loss after each completed 
#                 # epoch (1 epoch ^= data_train.df.shape[0]/data_train.batch_size
#                 # training steps => steps per epoch)
#                 val_epoch = int(data_train.df.shape[0] / data_train.batch_size)
                
#                 if data_train.num_iter % val_epoch == 0:
#                     # After each training epoch we will use the complete 
#                     # validation data set to calculate the error/accuracy on the
#                     # validation set:
#                     # loop through one validation data set epoch:
#                     # initialize dictionary which contains all validation 
#                     # losses:
#                     valid_loss_dict = dict.fromkeys(loss_dict, 0)
#                     valid_steps_per_epoch = int(data_valid.df.shape[0] / 
#                                                 data_valid.batch_size)
#                     for j in range(valid_steps_per_epoch):
#                         # DISCLAIMER: we do not run the opt_step here (on 
#                         # the validation data set) because we do not want to 
#                         # train our network on the validation set. Important 
#                         # for batch normalization and dropout 
#                         # (training -> False).
#                         # get validation data set (mini) batch:
#                         lst = iq.next_batch_sintel(deq=data_valid.dequeue(), 
#                                                    output_shape=image_shape,
#                                                    is_scale=False,
#                                                    is_flip=False,
#                                                    is_rotated=False,
#                                                    norm=norm)
#                         img_b_val, alb_b_val, shad_b_val, inv_b_val = lst

#                         # calculate the mean loss of this validation batch and 
#                         # sum it with the previous mean batch losses:
#                         fd_val = {x: img_b_val,
#                                   y_albedo_label: alb_b_val,
#                                   y_shading_label: shad_b_val,
#                                   invalid_px_mask: inv_b_val,
#                                   training: False}

#                         for key, val in loss_dict.items():
#                             # divide each loss loss by the iteration steps 
#                             # (steps_per_epoch) to get the mean val loss:
#                             mean_val = val / valid_steps_per_epoch
#                             valid_loss_dict[key] += sess.run(mean_val, 
#                                                              feed_dict=fd_val)
#                     # adding a mean loss summary op (for tensorboard):
#                     feed_dict_vl = {valid_dict[key]: valid_loss_dict[key] for \
#                                     key in valid_dict.keys()}
#                     val_loss_sums = sess.run(valid_sums_merged,
#                                              feed_dict=feed_dict_vl)
#                     summary_writer.add_summary(summary=val_loss_sums, 
#                                                global_step=global_step.eval())

#                     dur_time = time.time() - start_time
#                     dur_time = ghelp.get_time_format(time_in_sec=dur_time)
#                     dur_time = ghelp.time_tuple_to_str(time_tuple=dur_time)
#                     logger.info('step {} '.format(data_train.num_iter) +
#                                 '(epoch ' + 
#                                 '{}):'.format(data_train.completed_epochs + 1) +
#                                 ' mean validation losses:\n    ' +
#                                 ', '.join(['{}: {:.4f}'.format(it[0], it[1]) \
#                                            for it in valid_loss_dict.items()]) +
#                                 ' (ET: {}).'.format(dur_time))
#                     # reset timer to measure the displayed training steps:
#                     start_time = time.time()

#                 ################################################################

#                 if data_train.num_iter % save_step == 0:
#                     # save checkpoint files to disk:
#                     save_path = saver.save(sess, log_dir + 'tfmodel',
#                                            global_step=global_step.eval())
#                     s = sess.run(merge_train_summaries, feed_dict=feed_dict_tr)
#                     # adds a Summary protocol buffer to the event file 
#                     # (global_step: Number. Optional global step value to record
#                     # with the summary. Each stepp i is assigned to the 
#                     # corresponding summary parameter.)
#                     summary_writer.add_summary(summary=s, 
#                                                global_step=global_step.eval())
#                     dur_time = time.time() - start_time
#                     dur_time = ghelp.get_time_format(time_in_sec=dur_time)
#                     dur_time = ghelp.time_tuple_to_str(time_tuple=dur_time)
#                     logger.info('Saved data (step ' + 
#                                 '{}):\n'.format(data_train.num_iter) +
#                                 '    Checkpoint file written to: ' + 
#                                 '{} '.format(save_path) +
#                                 '(ET: {}).'.format(dur_time))
#                     # reset timer to measure the displayed training steps:
#                     start_time = time.time()

#             # end while loop when there are no elements left to dequeue:
#             except IndexError:
#                 end_total_time = time.time() - start_total_time
#                 end_total_time = ghelp.get_time_format(end_total_time)
#                 end_total_time = ghelp.time_tuple_to_str(time_tuple=end_total_time)
#                 logger.info('Training done... total training time: ' + 
#                             '{}.'.format(end_total_time))
#                 break

#     logger.info('Finished training.')


# In[3]:


def train_network_iiw(log_dir, data_dir, path_inference_graph, checkpoint_path,
                      restore_scope, image_shape, initial_learning_rate, 
                      loss_opt, lambda_loss, batch_size, num_epochs, 
                      display_step, save_step, nodes_name_dict, logger, 
                      dataset='iiw', is_sample=False, norm=True, 
                      plot_inference_graph=False):
    """
    :param log_dir: path to directory for saving summary/log files
    :type log_dir: str
    :param data_dir: path to directory with training and validation data (in 
        this directory the training and validation csv files have to be
        located.)
    :type data_dir: str
    :param path_inference_graph: path to inference graph (graph without 
        'training' ops)
    :type path_inference_graph: str
    :param checkpoint_path: path to model parameters (checkpoint files
        e.g. 'logs/2/tfmodel-5' or None)
    :type checkpoint_path: str
    :param restore_scope: a scope name which has to be defined to find 
        parameters to restore (e.g. 'vgg_16'). Usually this is needed to
        find a pre-trained (tf slim) model within another model.
    :type restore_scope: str
    :param image_shape: Shape of images that should be used for training 
        (shape of cnn input tensor)
    :type image_shape: list with len(image_shape)==3
    :param initial_learning_rate: hyper-parameters for initial learning rate
    :type initial_learning_rate: float
    :param loss_opt: loss function used for optimization 
        ('berhu', 'l2', 'l2_inv', 'l2_avg')
    :type loss_opt: str
    :param batch_size: nr of data which is put through the network before 
        updating it, as default use: 16, 32 or 64. 
        batch_size determines how many data samples are loaded in the memory 
        (be careful with memory space)
    :type batch_size: int
    :param num_epochs: nr of times the training process loops through the 
        complete training data set (how often is the tr set 'seen')
        if you have 1000 training examples, and your batch size is 500, then it
        will take 2 iterations to complete 1 epoch.
    :type num_epochs: int
    :param display_step: every display_step'th training iteration information is
        printed to stdout and file training.log in log_dir (default: 100)
    :type display_step: int
    :param save_step: every save_step'th training iteration a summary file is 
        written to log_dir and checkpoint files are saved
    :type save_step: int
    :param nodes_name_dict: dictionary that contains name of input, albedo and 
        shading output in form {'input': '', 
                                'output_albedo': '',
                                'output_shading': ''}
    :type nodes_name_dict: dict
    :param dataset: which dataset to use for training (default: 'sintel')
    :type dataset: str, must be \elem {'iiw', 'sintel'}
    :param is_sample: flag, if True only a smaller sample size is used for 
        training and validation (default: False).
    :type is_sample: boolean
    :param norm: flag, if True image pixels are scaled to range [0, 1]
        (default: True)
    :type norm: boolean
    :param plot_inference_graph: flag, True if inference graph should be 
        plotted (default: False).
    :type plot_inference_graph: boolean
    """
    ############################################################################
    logger.info('Training on images of shape: {}'.format(image_shape))
    logger.info('Training on [0, 1] normalized pixel values: {}'.format(norm))
    logger.info('Initial learning rate: {}'.format(initial_learning_rate))
    logger.info('Loss function used for optimization: {}'.format(loss_opt))
    logger.info('Batch size: {}'.format(batch_size))
    logger.info('# epochs: {}'.format(num_epochs))
    logger.info("Loss 'regularizer' (L_1/2 + lambda * M(W)HDL), " + 
                "lambda = {}".format(lambda_loss))
    logger.info('Write summary and checkpoints to file (in directory ' +
                '{}) every '.format(log_dir) +
                '{} iterations.'.format(save_step))

    # load meta graph (inference graph)
    # how to work with restored models:
    # https://www.tensorflow.org/programmers_guide/meta_graph
    # http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
    saver_restore = tf.train.import_meta_graph(path_inference_graph, 
                                               clear_devices=True)

    logger.debug('Restored inference graph from\n' +
                 '    {}'.format(path_inference_graph))
    variables_to_restore = slim.get_variables_to_restore(include=[restore_scope],
                                                         exclude=None)
    logger.info('# of parameters that can be restored: ' + 
                '{}.'.format(len(variables_to_restore)))

    ############################################################################

    # save default graph in variable:
    graph = tf.get_default_graph()
    if plot_inference_graph:
        # plot imported inference graph:
        plt_help.show_graph(graph.as_graph_def())

    # lets get the input
    x = graph.get_tensor_by_name(name=nodes_name_dict['input'])

    # bool variable that indicates if we are in training mode (training=True) or
    # valid/test mode (training=False) this indicator is important if dropout 
    # or/and batch normalization is used.
    try:
        # try importing training node (is needed for models that use batch 
        # normalization etc.)
        training = graph.get_tensor_by_name(name='is_training_1:0')
        logger.debug('Was able to catch is_training node!')
    except KeyError:
        # elsewise just define a placeholder wich is used as dummy variable
        # and won't be used later:
        training = tf.placeholder(dtype=tf.bool, name='is_training_1')
        logger.debug('Initialized is_training node.')

    # define new global step variable:
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # get the last global training step if we continue training:
    try:
        last_global_step = int(checkpoint_path.split('/')[-1].split('-')[-1])
    except (ValueError, AttributeError):
        last_global_step = 0
    
    # get graph output nodes:
    y_albedo_pred = graph.get_tensor_by_name(name=nodes_name_dict['output_albedo'])
    y_shading_pred = graph.get_tensor_by_name(name=nodes_name_dict['output_shading'])
    # y_albedo_pred = tf.clip_by_value(t=y_albedo, clip_value_min=0, 
    #                                  clip_value_max=1, 
    #                                  name='0_1_clipping_albedo')
    # y_shading_pred = tf.clip_by_value(t=y_shading, clip_value_min=0,
    #                                   clip_value_max=1, 
    #                                   name='0_1_clipping_shading')

    ############################################################################
    ############################################################################

    with tf.name_scope('loss'):
        if loss_opt not in ('l1_mhdl', 'l1_mwhdl', 'l2_mhdl', 
                            'l2_mwhdl'):
            raise ValueError('{} is not a valid loss '.format(loss_opt) + 
                             'function. Set parameter loss_opt to one of the ' +
                             "following: ('l1_mhdl', 'l1_mwhdl', " + 
                             "l2_mhdl', 'l2_mwhdl')")
            
    
        point1 = tf.placeholder(dtype=tf.int32, name='point1')
        point2 = tf.placeholder(dtype=tf.int32, name='point2')
        # get the human darker labels:
        human_labels = tf.placeholder(dtype=tf.int32, name='human_labels')
        # list of weights/darker scores:
        darker_weights = tf.placeholder(dtype=tf.float32, name='darker_weights')

        losses = chlp.iiw_loss_fct(input_image=x,
                                   prediction_albedo=y_albedo_pred,
                                   prediction_shading=y_shading_pred,
                                   albedo_comp_point1=point1,
                                   albedo_comp_point2=point2,
                                   albedo_comp_human_labels=human_labels,
                                   albedo_comp_weights=darker_weights,
                                   albedo_comp_delta=0.1,
                                   lambda_=lambda_loss)

        loss_dict = {'l1_mhdl': losses[0],
                     'l1_mwhdl': losses[1],
                     'l2_mhdl': losses[2],
                     'l2_mwhdl': losses[3],
                     'l1': losses[4],
                     'l2': losses[5],
                     'mhdl': losses[6],
                     'mwhdl': losses[7]}
        
        loss = loss_dict[loss_opt]

    logger.debug('Defined training losses.')

    ############################################################################

    # Use an AdamOptimizer to train the network:
    with tf.name_scope('optimization'):
        optimizer = tf.train.AdamOptimizer(learning_rate=initial_learning_rate)
        
        # use slim optimizaton op only if a batch normalization is in the 
        # tf graph.
        use_slim_train_opt = False
        for v in graph.as_graph_def().node:
            if 'batchnorm' in v.name.lower():
                use_slim_train_opt = True
                break
        # Many networks utilize modules, like BatchNorm, that require 
        # performing a series of non-gradient updates during training. 
        # slim.learning.create_train_op allows a user to pass in a list of 
        # update_ops to call along with the gradient updates.
        #   train_op = slim.learning.create_train_op(total_loss, optimizer,
        #                                            update_ops)
        # By default, slim.learning.create_train_op includes all update ops 
        # that are part of the `tf.GraphKeys.UPDATE_OPS` collection.
        # Additionally, TF-Slim's slim.batch_norm function adds the moving mean
        # and moving variance updates to this collection. Consequently, users 
        # who want to use slim.batch_norm will not need to take any additional
        # steps in order to have the moving mean and moving variance updates be
        # computed.
        # (see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/learning.py)
        if use_slim_train_opt:
            opt_step = slim.learning.create_train_op(total_loss=loss, 
                                                     optimizer=optimizer,
                                                     global_step=global_step) 
            logger.debug('Using slim optimizaton op because a batch ' +
                         'normalization is apparently in the tf graph.')
        else:
            opt_step = optimizer.minimize(loss, global_step=global_step)
            logger.debug('Using default tf optimizaton op because there is ' +
                         'apparently no batch normalization in the tf ' +
                         'graph.')

    logger.debug('Defined optimization method.')
    
    
    ############################################################################

    # to get every summary defined above we merge them to get one target:
    merge_train_summaries = tf.summary.merge_all()
    # define a FileWriter op which writes summaries defined above to disk:
    summary_writer = tf.summary.FileWriter(log_dir)
    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(max_to_keep=None)

    ############################################################################

    # introduce some validation set specific summaries
    # These summaries need to be defined blow the function 
    # merge_train_summaries = tf.summary.merge_all()
    # because the validation set summaries are added to the summary writer at 
    # different times. If they had been summarized with the training summaries 
    # they would have to be defined at times where merge_train_summaries are 
    # added to the summary writer
    with tf.name_scope('loss/valid/'):
        valid_dict = {key: tf.placeholder(dtype=tf.float32, name=key) for                       key in loss_dict.keys()}
        valid_summaries = [tf.summary.scalar(name=key, tensor=val) for                            key, val in valid_dict.items()]
        valid_sums_merged = tf.summary.merge(inputs=valid_summaries,
                                             collections=None, name=None)
    logger.debug('Defined validation losses.')
    logger.debug('Finished building training graph.')
    logger.info('Total parameters of network: ' +
                '{}'.format(nnhelp.network_params()))

    ############################################################################
    ############################################################################

    # import data:
    if is_sample:
        sample = 'sample_'
    else:
        sample = ''
    
    file = sample + 'data_iiw_train.csv'
    df_train = pd.read_csv(filepath_or_buffer=data_dir + file, 
                           sep=',', header=None,
                           names=['img', 'json_label'])
    # complete image paths:
    df_train = data_dir + df_train

    # instantiate a data queue for feeding data in (mini) batches to cnn:
    data_train = iq.DataQueue(df=df_train, batch_size=batch_size,
                              num_epochs=num_epochs)

    logger.debug('Imported {} training data '.format(dataset) + 
                 '(#: {}) '.format(data_train.df.shape[0]) + 
                 'from\n    {}'.format(data_dir + file))
    
    
    file = sample + 'data_iiw_valid.csv'
    df_valid = pd.read_csv(filepath_or_buffer=data_dir + file, 
                           sep=',', header=None,
                           names=['img', 'json_label'])
    # complete image paths:
    df_valid = data_dir + df_valid
    # instantiate a data queue for feeding data in (mini) batches to cnn:
    data_valid = iq.DataQueue(df=df_valid, batch_size=batch_size,
                              num_epochs=num_epochs)
    logger.debug('Imported {} validation data '.format(dataset) +
                 '(#: {}) '.format(data_valid.df.shape[0]) + 
                 'from\n    {}'.format(data_dir + file))

    ############################################################################
    ############################################################################

    logger.info('Start training:')
    # Initialization:
    # Op that initializes global variables in the graph:
    init_global = tf.global_variables_initializer()
    # Op that initializes local variables in the graph:
    init_local = tf.local_variables_initializer()

    config = tf.ConfigProto(device_count = {'GPU': 1},
                            intra_op_parallelism_threads=3
    #                        allow_soft_placement = True,
    #                        log_device_placement=False
                           )
    with tf.Session(config=config) as sess: 
        
        ########################################################################
        
        # initialize all variables:
        sess.run([init_global, init_local])
        
        # assign the last true global step to to global step:
        sess.run(global_step.assign(last_global_step))
        logger.info('Assigned last global training step: ' +
                    '{}'.format(global_step.eval()))

        if checkpoint_path:
            try:
                # restore saved model parameters (weights, biases, etc):
                saver_restore.restore(sess, checkpoint_path)
                logger.info('Restoring parameters from ' +
                            '{}'.format(checkpoint_path))
            except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError):
                # in the worst case parameters are loaded twice.
                # restore the parameters:
                init_fn = slim.assign_from_checkpoint_fn(checkpoint_path,
                                                         variables_to_restore)
                init_fn(sess)
                logger.info('Restoring parameters from ' +
                            '{}'.format(checkpoint_path))
        # Adds a Graph to the event file.
        # create summary that give output (TensorFlow op that output protocol 
        # buffers containing 'summarized' data) of the built Tensorflow graph:
        summary_writer.add_graph(sess.graph)

        # start timer for total training time:
        start_total_time = time.time()
        # set timer to measure the displayed training steps:
        start_time = start_total_time

        ########################################################################
        
        # Training:
        # train loop
        # train until all data is processed (queue empty),
        # number of iterations depends on number of data, number of epochs and 
        # batch size:
        iter_start = data_train.iter_left
        logger.info('For training it takes {}\n'.format(iter_start) +
                    '    (= # data / batch_size * epochs) iterations to loop ' +
                    'through {} samples of\n  '.format(data_train.df.shape[0]) +
                    '  training data over {} '.format(data_train.num_epochs) +
                    'epochs summarized in batches of size ' + 
                    '{}.\n'.format(data_train.batch_size) +
                    '    So, there are # data / batch_size = ' +
                    '{}'.format(int(data_train.df.shape[0]/data_train.batch_size))+
                    ' iterations per epoch.')

        while data_train.iter_left >= 0:
            try:
                # take a (mini) batch of the training data:
                deq_train = data_train.dequeue()
                lst = iq.next_batch_iiw(deq=deq_train,
                                        output_shape=image_shape,
                                        norm=norm)
                (df_whdr, imgs_b, imgs_original_b, 
                 js_label_b, js_originial_label_b) = lst
                fd_tr = {x: imgs_b,
                         point1: df_whdr[['batch_nr', 'y1', 'x1']].values,
                         point2: df_whdr[['batch_nr', 'y2', 'x2']].values,
                         human_labels: df_whdr['darker'].values,
                         darker_weights: df_whdr['darker_score'].values,
                         training: True}
                # run optimization step:
                sess.run(opt_step, feed_dict=fd_tr)

                ################################################################

                # report training set accuracy every display_step-th step:
                if (data_train.num_iter) % display_step == 0:
                    # console output:
                    train_loss_dict = {}
                    for key, val in loss_dict.items():
                        train_loss_dict[key] = sess.run(val, 
                                                        feed_dict=fd_tr)

                    dur_time = time.time() - start_time
                    dur_time = ghelp.get_time_format(time_in_sec=dur_time)
                    dur_time = ghelp.time_tuple_to_str(time_tuple=dur_time)
                    logger.info('step {}: '.format(data_train.num_iter) +
                                'training ({}) loss '.format(loss_opt) +
                                '{:.4f}'.format(train_loss_dict[loss_opt]) + 
                                '\n    (' +
                                ', '.join(['{}: {:.4f}'.format(it[0], it[1]) \
                                           for it in train_loss_dict.items() \
                                           if it[0]!=loss_opt]) +
                                ', ET: {}).'.format(dur_time))
                    # reset timer to measure the displayed training steps:
                    start_time = time.time()

                ################################################################

                # Validation:
                # display validation set accuracy and loss after each completed 
                # epoch (1 epoch ^= data_train.df.shape[0]/data_train.batch_size
                # training steps => steps per epoch)
                val_epoch = int(data_train.df.shape[0] / data_train.batch_size)
                
                if data_train.num_iter % val_epoch == 0:
                    # After each training epoch we will use the complete 
                    # validation data set to calculate the error/accuracy on the
                    # validation set:
                    # loop through one validation data set epoch:
                    # initialize dictionary which contains all validation 
                    # losses:
                    valid_loss_dict = dict.fromkeys(loss_dict, 0)
                    valid_steps_per_epoch = int(data_valid.df.shape[0] / 
                                                data_valid.batch_size)
                    for j in range(valid_steps_per_epoch):
                        # DISCLAIMER: we do not run the opt_step here (on 
                        # the validation data set) because we do not want to 
                        # train our network on the validation set. Important 
                        # for batch normalization and dropout 
                        # (training -> False).
                        # get validation data set (mini) batch:
                        lst = iq.next_batch_iiw(deq=data_valid.dequeue(),
                                                output_shape=image_shape,
                                                norm=norm)
                        (df_whdr_val, imgs_b_val, imgs_original_b_val, 
                         js_label_b_val, js_originial_label_b_val) = lst

                        # calculate the mean loss of this validation batch and 
                        # sum it with the previous mean batch losses:
                        fd_val = {x: imgs_b_val,
                                  point1: df_whdr_val[['batch_nr', 
                                                       'y1', 'x1']].values,
                                  point2: df_whdr_val[['batch_nr',
                                                       'y2', 'x2']].values,
                                  human_labels: df_whdr_val['darker'].values,
                                  darker_weights: df_whdr_val['darker_score'].values,
                                  training: False}

                        for key, val in loss_dict.items():
                            # divide each loss loss by the iteration steps 
                            # (steps_per_epoch) to get the mean val loss:
                            mean_val = val / valid_steps_per_epoch
                            valid_loss_dict[key] += sess.run(mean_val, 
                                                             feed_dict=fd_val)
                    # adding a mean loss summary op (for tensorboard):
                    feed_dict_vl = {valid_dict[key]: valid_loss_dict[key] for                                     key in valid_dict.keys()}
                    val_loss_sums = sess.run(valid_sums_merged,
                                             feed_dict=feed_dict_vl)
                    summary_writer.add_summary(summary=val_loss_sums, 
                                               global_step=global_step.eval())
                    dur_time = time.time() - start_time
                    dur_time = ghelp.get_time_format(time_in_sec=dur_time)
                    dur_time = ghelp.time_tuple_to_str(time_tuple=dur_time)
                    logger.info('step {} '.format(data_train.num_iter) +
                                '(epoch ' + 
                                '{}):'.format(data_train.completed_epochs + 1) +
                                ' mean validation losses:\n    ' +
                                ', '.join(['{}: {:.4f}'.format(it[0], it[1]) \
                                           for it in valid_loss_dict.items()]) +
                                ' (ET: {}).'.format(dur_time))
                    # reset timer to measure the displayed training steps:
                    start_time = time.time()

                ################################################################

                if data_train.num_iter % save_step == 0:
                    # save checkpoint files to disk:
                    save_path = saver.save(sess, log_dir + 'tfmodel',
                                           global_step=global_step.eval())
                    
                    s = sess.run(merge_train_summaries, feed_dict=fd_tr)
                    # adds a Summary protocol buffer to the event file 
                    # (global_step: Number. Optional global step value to record
                    # with the summary. Each stepp i is assigned to the 
                    # corresponding summary parameter.)
                    summary_writer.add_summary(summary=s, 
                                               global_step=global_step.eval())
                    dur_time = time.time() - start_time
                    dur_time = ghelp.get_time_format(time_in_sec=dur_time)
                    dur_time = ghelp.time_tuple_to_str(time_tuple=dur_time)
                    logger.info('Saved data (step ' + 
                                '{}):\n'.format(data_train.num_iter) +
                                '    Checkpoint file written to: ' + 
                                '{} '.format(save_path) +
                                '(ET: {}).'.format(dur_time))
                    # reset timer to measure the displayed training steps:
                    start_time = time.time()

            # end while loop when there are no elements left to dequeue:
            except IndexError:
                end_total_time = time.time() - start_total_time
                end_total_time = ghelp.get_time_format(end_total_time)
                end_total_time = ghelp.time_tuple_to_str(time_tuple=end_total_time)
                logger.info('Training done... total training time: ' + 
                            '{}.'.format(end_total_time))

                break

    logger.info('Finished training.')


# In[4]:


def train_network(dataset, **kwargs):
    # create logger (write to file and stdout):
    logger = ghelp.create_logger(filename=kwargs['log_dir'] + 'training.log')
    logger.debug('Python version: \n    ' + sys.version + 
                 '\n    Tensorflow version: ' + tf.__version__)
    logger.info('Parameter summary: \n' + 
                "\n".join("{}: {}".format(k, v) for k, v in params.items()))
    
    if dataset=='iiw':
        return train_network_iiw(logger=logger, **kwargs)
    elif dataset=='sintel':
        return train_network_sintel(logger=logger, **kwargs)
    elif dataset=='sintel_upproject':
        return train_network_sintel_upproject(logger=logger, **kwargs)
    else:
        raise ValueError("Invalid dataset! Enter one of the " +
                         "following: ('iiw', 'sintel')")


# In[5]:


# # sintel_slim_resnet_v1_50_upproject:
# # this model uses a pre-trained resnet_50 as encoder and a decoder which
# # constists of 1 scales of up-projection blocks:
# nodes_name_dict = {}
# # download checkpoint files:
# url = "http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz"
# checkpoints_dir = './models/slim/checkpoints'
# print('If not available download resnet_v1_50 ckpt files to ' + checkpoints_dir)
# download.maybe_download_and_extract(url=url, 
#                                     download_dir=checkpoints_dir,
#                                     print_download_progress=True)
# params = {'log_dir': 'logs/sintel/slim_resnet_v1_50_upproject/test/',
#           'data_dir': '/media/sdb/udo/data/',
#           'path_inference_graph': '',
#           'checkpoint_path': 'models/slim/checkpoints/resnet_v1_50.ckpt',
#           'restore_scope': 'resnet_v1_50',
#           'image_shape': [320, 320, 3],
#           'initial_learning_rate': 5e-4,  # hyper param
#           'loss_opt': 'l1',
#           'batch_size': 16,  # hyper param
#           'num_epochs': 10,  # hyper param 
#           'display_step': 2,
#           'save_step': 5,
#           'nodes_name_dict': nodes_name_dict,
#           'dataset': 'sintel_upproject',
#           'is_sample': False,
#           'norm': True,
#           'plot_inference_graph': False
#          }


# In[6]:


################################################################################
################################################################################
# sintel dataset parameters:

# # sintel_narihira2015:
# # this model is the implementation of the narihira2015 network.
# # 2 scales, encoder is basically AlexNet, decoder consists mainly of
# # deconvolution functions as up-scaling functions. This model implementation
# # does not have any pre-trained weights.
# # Narihira et al recommend using multiples of 13 for spatial image input sizes
# # (eg: [32 * 13, 32 * 13, 3])
# nodes_name_dict = {'input': 'input:0',
#                    'output_albedo': 'deconv_s2out_albedo/BiasAdd:0',
#                    'output_shading': 'deconv_s2out_shading/BiasAdd:0'}

# params = {'log_dir': 'logs/sintel/narihira2015/test/',  # path to summary files
#           'data_dir': '/media/sdb/udo/data/',
#           'path_inference_graph': 'models/narihira2015/tfmodel_inference.meta',
#           'checkpoint_path': None, # e.g. 'logs/2/tfmodel-5' or None
#           'restore_scope': None,
#           'image_shape': [320, 320, 3],
#           'initial_learning_rate': 5e-4,  # hyper param
#           'loss_opt': 'l1',
#           'batch_size': 16,  # hyper param
#           'num_epochs': 100,  # hyper param 
#           'display_step': 2,
#           'save_step': 100,
#           'nodes_name_dict': nodes_name_dict,
#           'dataset': 'sintel',
#           'is_sample': False,
#           'norm': True,
#           'plot_inference_graph': False
#          }


# # sintel_slim_vgg16_narihira2015:
# # this model uses a pre-trained vgg16 as encoder and a decoder which
# # constists of 2 scales (main upscaling function: deconvolution). 
# # This decoder has basically the proposed structure of narihira2015:
# nodes_name_dict = {'input': 'input:0',
#                    'output_albedo': 'scale2/deconv6_s2_albedo/BiasAdd:0',
#                    'output_shading': 'scale2/deconv6_s2_shading/BiasAdd:0'}
# # download checkpoint files:
# url = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"
# checkpoints_dir = './models/slim/checkpoints'
# print('If not available download vgg16 ckpt files to ' + checkpoints_dir)
# download.maybe_download_and_extract(url=url, 
#                                     download_dir=checkpoints_dir,
#                                     print_download_progress=True)
# params = {'log_dir': 'logs/sintel/slim_vgg16_narihira2015/test/',
#           'data_dir': '/media/sdb/udo/data/',
#           'path_inference_graph': 'models/slim/graphs/vgg16_narihira2015/tfmodel_inference.meta',
#           'checkpoint_path': 'models/slim/checkpoints/vgg_16.ckpt',
#           'restore_scope': 'vgg_16',
#           'image_shape': [320, 320, 3],
#           'initial_learning_rate': 5e-4,  # hyper param
#           'loss_opt': 'l1',
#           'batch_size': 16,  # hyper param
#           'num_epochs': 100,  # hyper param 
#           'display_step': 20,
#           'save_step': 100,
#           'nodes_name_dict': nodes_name_dict,
#           'dataset': 'sintel',
#           'is_sample': False,
#           'norm': True,
#           'plot_inference_graph': False
#          }


# # sintel_slim_vgg16_deconv_decoder:
# # this model uses a pre-trained vgg16 as encoder and a decoder
# # which basically consists of deconvolution functions as upscaling functions
# # (it uses only 1 scale):
# nodes_name_dict = {'input': 'input:0',
#                    'output_albedo': 'scale2/deconv6_s2_albedo/BiasAdd:0',
#                    'output_shading': 'scale2/deconv6_s2_shading/BiasAdd:0'}
# # download checkpoint files:
# url = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"
# checkpoints_dir = './models/slim/checkpoints'
# print('If not available download vgg16 ckpt files to ' + checkpoints_dir)
# download.maybe_download_and_extract(url=url, 
#                                     download_dir=checkpoints_dir,
#                                     print_download_progress=True)
# params = {'log_dir': 'logs/sintel/slim_vgg16_deconv_decoder/test/',
#           'data_dir': '/media/sdb/udo/data/',
#           'path_inference_graph': 'models/slim/graphs/vgg16/tfmodel_inference.meta',
#           'checkpoint_path': 'models/slim/checkpoints/vgg_16.ckpt',
#           'restore_scope': 'vgg_16',
#           'image_shape': [320, 320, 3],
#           'initial_learning_rate': 5e-4,  # hyper param
#           'loss_opt': 'l1',
#           'batch_size': 16,  # hyper param
#           'num_epochs': 100,  # hyper param 
#           'display_step': 2,
#           'save_step': 100,
#           'nodes_name_dict': nodes_name_dict,
#           'dataset': 'sintel',
#           'is_sample': False,
#           'norm': True,
#           'plot_inference_graph': False
#          }


# # sintel_slim_resnet_v1_50_deconv_decoder:
# # this model uses a pre-trained resnet_50 as encoder and a decoder 
# # which basically consists of deconvolution functions as upscaling functions
# # (it uses only 1 scale):
# nodes_name_dict = {'input': 'input:0',
#                    'output_albedo': 'decoder/deconv7_albedo/BiasAdd:0',
#                    'output_shading': 'decoder/deconv7_shading/BiasAdd:0'}
# # download checkpoint files:
# url = "http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz"
# checkpoints_dir = './models/slim/checkpoints'
# print('If not available download resnet_v1_50 ckpt files to ' + checkpoints_dir)
# download.maybe_download_and_extract(url=url, 
#                                     download_dir=checkpoints_dir,
#                                     print_download_progress=True)
# params = {'log_dir': 'logs/sintel/slim_resnet_v1_50_deconv_decoder/test/',
#           'data_dir': '/media/sdb/udo/data/',
#           'path_inference_graph': 'models/slim/graphs/resnet_v1_50/tfmodel_inference.meta',
#           'checkpoint_path': 'models/slim/checkpoints/resnet_v1_50.ckpt',
#           'restore_scope': 'resnet_v1_50',
#           'image_shape': [320, 320, 3],
#           'initial_learning_rate': 5e-4,  # hyper param
#           'loss_opt': 'l1',
#           'batch_size': 16,  # hyper param
#           'num_epochs': 100,  # hyper param 
#           'display_step': 20,
#           'save_step': 100,
#           'nodes_name_dict': nodes_name_dict,
#           'dataset': 'sintel',
#           'is_sample': False,
#           'norm': True,
#           'plot_inference_graph': False
#          }


# # sintel_slim_resnet_v1_50_narihira2015:
# # this model uses a pre-trained resnet_50 as encoder and a decoder which
# # constists of 2 scales (main upscaling function: deconvolution). 
# # This decoder has basically the proposed structure of narihira2015:
# nodes_name_dict = {'input': 'input:0',
#                    'output_albedo': 'decoder/deconv7_albedo/BiasAdd:0',
#                    'output_shading': 'decoder/deconv7_shading/BiasAdd:0'}
# # download checkpoint files:
# url = "http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz"
# checkpoints_dir = './models/slim/checkpoints'
# print('If not available download resnet_v1_50 ckpt files to ' + checkpoints_dir)
# download.maybe_download_and_extract(url=url, 
#                                     download_dir=checkpoints_dir,
#                                     print_download_progress=True)
# params = {'log_dir': 'logs/sintel/slim_resnet_v1_50_narihira2015/test/',
#           'data_dir': '/media/sdb/udo/data/',
#           'path_inference_graph': 'models/slim/graphs/resnet_v1_50_narihira2015/tfmodel_inference.meta',
#           'checkpoint_path': 'models/slim/checkpoints/resnet_v1_50.ckpt',
#           'restore_scope': 'resnet_v1_50',
#           'image_shape': [320, 320, 3],
#           'initial_learning_rate': 5e-4,  # hyper param
#           'loss_opt': 'l1',
#           'batch_size': 16,  # hyper param
#           'num_epochs': 100,  # hyper param 
#           'display_step': 2,
#           'save_step': 100,
#           'nodes_name_dict': nodes_name_dict,
#           'dataset': 'sintel',
#           'is_sample': False,
#           'norm': True,
#           'plot_inference_graph': False
#          }

# # sintel_slim_resnet_v1_50_narihira2015_reduced:
# # this model uses a pre-trained resnet_50 as encoder and a decoder which
# # constists of 2 scales (main upscaling function: deconvolution). 
# # This decoder has basically the proposed structure of narihira2015:
# nodes_name_dict = {'input': 'input:0',
#                    'output_albedo': 'decoder/deconv7_albedo/BiasAdd:0',
#                    'output_shading': 'decoder/deconv7_shading/BiasAdd:0'}
# # download checkpoint files:
# url = "http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz"
# checkpoints_dir = './models/slim/checkpoints'
# print('If not available download resnet_v1_50 ckpt files to ' + checkpoints_dir)
# download.maybe_download_and_extract(url=url, 
#                                     download_dir=checkpoints_dir,
#                                     print_download_progress=True)
# params = {'log_dir': 'logs/sintel/slim_resnet_v1_50_narihira2015_reduced/test/',
#           'data_dir': '/media/sdb/udo/data/',
#           'path_inference_graph': 'models/slim/graphs/resnet_v1_50_narihira2015_reduced/tfmodel_inference.meta',
#           'checkpoint_path': 'models/slim/checkpoints/resnet_v1_50.ckpt',
#           'restore_scope': 'resnet_v1_50',
#           'image_shape': [320, 320, 3],
#           'initial_learning_rate': 5e-4,  # hyper param
#           'loss_opt': 'l1',
#           'batch_size': 16,  # hyper param
#           'num_epochs': 100,  # hyper param 
#           'display_step': 20,
#           'save_step': 100,
#           'nodes_name_dict': nodes_name_dict,
#           'dataset': 'sintel',
#           'is_sample': False,
#           'norm': True,
#           'plot_inference_graph': False
#          }


# sintel_slim_resnet_v1_50_deconv_decoder_reduced:
# this model uses a pre-trained resnet_50 as encoder and a decoder 
# which basically consists of deconvolution functions as upscaling functions
# (it uses only 1 scale):
nodes_name_dict = {'input': 'input:0',
                   'output_albedo': 'decoder/deconv7_albedo/BiasAdd:0',
                   'output_shading': 'decoder/deconv7_shading/BiasAdd:0'}
# download checkpoint files:
url = "http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz"
checkpoints_dir = './models/slim/checkpoints'
print('If not available download resnet_v1_50 ckpt files to ' + checkpoints_dir)
download.maybe_download_and_extract(url=url, 
                                    download_dir=checkpoints_dir,
                                    print_download_progress=True)
params = {'log_dir': 'logs/sintel/slim_resnet_v1_50_deconv_decoder_reduced/1/',
          'data_dir': '/media/sdb/udo/data/',
          'path_inference_graph': 'models/slim/graphs/resnet_v1_50_reduced/tfmodel_inference.meta',
          'checkpoint_path': 'models/slim/checkpoints/resnet_v1_50.ckpt',
          'restore_scope': 'resnet_v1_50',
          'image_shape': [320, 320, 3],
          'initial_learning_rate': 5e-4,  # hyper param
          'loss_opt': 'l1',
          'batch_size': 16,  # hyper param
          'num_epochs': 100,  # hyper param 
          'display_step': 20,
          'save_step': 100,
          'nodes_name_dict': nodes_name_dict,
          'dataset': 'sintel',
          'is_sample': False,
          'norm': True,
          'plot_inference_graph': False
         }

################################################################################
################################################################################
# iiw dataset parameters:

# # iiw_slim_resnet_v1_50_deconv_decoder:
# # this model uses a pre-trained resnet_50 as encoder and  a decoder 
# # which basically consists of deconvolution functions as upscaling functions
# # (it uses only 1 scale)
# nodes_name_dict = {'input': 'input:0',
#                    'output_albedo': 'decoder/deconv7_albedo/BiasAdd:0',
#                    'output_shading': 'decoder/deconv7_shading/BiasAdd:0'}
# # download checkpoint files:
# url = "http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz"
# checkpoints_dir = './models/slim/checkpoints'
# print('If not available download resnet_v1_50 ckpt files to ' + checkpoints_dir)
# download.maybe_download_and_extract(url=url, 
#                                     download_dir=checkpoints_dir,
#                                     print_download_progress=True)
# params = {'log_dir': 'logs/iiw/slim_resnet_v1_50_deconv_decoder/norm_l1_10mhdl/test/',
#           'data_dir': '/media/sdb/udo/data/',
#           'path_inference_graph': 'models/slim/graphs/resnet_v1_50/tfmodel_inference.meta',
#           'checkpoint_path': 'models/slim/checkpoints/resnet_v1_50.ckpt',
#           'restore_scope': 'resnet_v1_50',
#           'image_shape': [320, 320, 3],
#           'initial_learning_rate': 5e-4,  # hyper param
#           'loss_opt': 'l1_mhdl',
#           'lambda_loss': 1.0,
#           'batch_size': 16,  # hyper param
#           'num_epochs': 10,  # hyper param 
#           'display_step': 2,
#           'save_step': 100,
#           'nodes_name_dict': nodes_name_dict,
#           'dataset': 'iiw',  # in iiw dataset min image height/width = 340 px
#           'is_sample': False,
#           'norm': True,
#           'plot_inference_graph': False
#          }


# # iiw_slim_vgg16_deconv_decoder:
# # this model uses a pre-trained vgg16 as encoder and a decoder
# # which basically consists of deconvolution functions as upscaling functions
# # (it uses only 1 scale):
# nodes_name_dict = {'input': 'input:0',
#                    'output_albedo': 'scale2/deconv6_s2_albedo/BiasAdd:0',
#                    'output_shading': 'scale2/deconv6_s2_shading/BiasAdd:0'}
# # download checkpoint files:
# url = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"
# checkpoints_dir = './models/slim/checkpoints'
# print('If not available download vgg16 ckpt files to ' + checkpoints_dir)
# download.maybe_download_and_extract(url=url, 
#                                     download_dir=checkpoints_dir,
#                                     print_download_progress=True)
# params = {'log_dir': 'logs/iiw/slim_vgg16_deconv_decoder/norm_l1_10mhdl/test/',
#           'data_dir': '/media/sdb/udo/data/',
#           'path_inference_graph': 'models/slim/graphs/vgg16/tfmodel_inference.meta',
#           'checkpoint_path': 'models/slim/checkpoints/vgg_16.ckpt',
#           'restore_scope': 'vgg_16',
#           'image_shape': [320, 320, 3],
#           'initial_learning_rate': 5e-4,  # hyper param
#           'loss_opt': 'l1_mhdl',
#           'lambda_loss': 1.0,
#           'batch_size': 16,  # hyper param
#           'num_epochs': 20,  # hyper param 
#           'display_step': 20,
#           'save_step': 100,
#           'nodes_name_dict': nodes_name_dict,
#           'dataset': 'iiw',
#           'is_sample': False,
#           'norm': True,
#           'plot_inference_graph': False
#          }


# # iiw_slim_vgg16_narihira2015:
# # this model uses a pre-trained vgg16 as encoder and a decoder which
# # constists of 2 scales (main upscaling function: deconvolution). 
# # This decoder has basically the proposed structure of narihira2015:
# nodes_name_dict = {'input': 'input:0',
#                    'output_albedo': 'scale2/deconv6_s2_albedo/BiasAdd:0',
#                    'output_shading': 'scale2/deconv6_s2_shading/BiasAdd:0'}
# # download checkpoint files:
# url = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"
# checkpoints_dir = './models/slim/checkpoints'
# print('If not available download vgg16 ckpt files to ' + checkpoints_dir)
# download.maybe_download_and_extract(url=url, 
#                                     download_dir=checkpoints_dir,
#                                     print_download_progress=True)
# params = {'log_dir': 'logs/iiw/slim_vgg16_narihira2015/norm_l1_10mhdl/test/',
#           'data_dir': '/media/sdb/udo/data/',
#           'path_inference_graph': 'models/slim/graphs/vgg16_narihira2015/tfmodel_inference.meta',
#           'checkpoint_path': 'models/slim/checkpoints/vgg_16.ckpt',
#           'restore_scope': 'vgg_16',
#           'image_shape': [320, 320, 3],
#           'initial_learning_rate': 5e-4,  # hyper param
#           'loss_opt': 'l1_mhdl',
#           'lambda_loss': 1.0,
#           'batch_size': 16,  # hyper param
#           'num_epochs': 20,  # hyper param 
#           'display_step': 20,
#           'save_step': 100,
#           'nodes_name_dict': nodes_name_dict,
#           'dataset': 'iiw',
#           'is_sample': False,
#           'norm': True,
#           'plot_inference_graph': False
#          }

################################################################################
################################################################################


# In[ ]:


train_network(**params)


# In[ ]:


# !tensorboard --logdir /logs/3

