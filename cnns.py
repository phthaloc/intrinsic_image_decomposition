
# coding: utf-8

# # Import inference graph and extend it to a training graph:
# 
# To extend the graph, we have to add:
#     - input data structure
#     - loss function
#     - optimization op
#     
# To build the graph it is necessary to know the node names of all relevant 
# layers, placeholders etc. for restoring the model later.
# 
# pay ATTENTION to:
#     the imported model must be in the same graph as the nodes which are 
#     added later
#     -> load the model first to the default graph, then add further ops
#     
# also test to input input-images of different sizes (multiples of 32px). 
# It might not work because input placeholder is defined fix.
# perhaps input node should not be saved inside the model?!
# 
# output of model has name (deconv_s2out_shading/BiasAdd:0 and 
# deconv_s2out_albedo/BiasAdd:0). 
# simpler names?!
# 
# 
# To plot all graphs directly in this notebook, run jupyter form terminal like 
# this:
#     jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000000

# In[1]:


import os   
import sys
sys.path.append('./util')
import time
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import input_queues as iq
import cnn_model
import plot_helpers as plt_help
import general_helpers as ghelp
import nn_helpers as nnhelp
import cnn_helpers as cnnhelp

# make only 'gpu:0' visible, so that only one gpu is used not both, see also
# https://github.com/tensorflow/tensorflow/issues/5066
# https://github.com/tensorflow/tensorflow/issues/3644#issuecomment-237631171
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

LOGS_PATH = 'logs/3/'  # path to summary files
# create logger (write to file and stdout):
logger = ghelp.create_logger(filename=LOGS_PATH + 'training.log')

logger.debug('Python version: \n    ' + sys.version + 
             '\n    Tensorflow version: ' + tf.__version__)


# In[2]:


################################################################################
################################################################################

# data path constants:
# DATA_DIR = '../data/mnist/'
DATA_DIR = 'data/'
# DATA_DIR = '/usr/udo/data/'

path_inference_graph_dict = {'narihira2015': 'models/narihira2015/' +
                                             'tfmodel_inference.meta'}
path_inference_graph = path_inference_graph_dict['narihira2015']
logger.info('Path to infrence graph:\n    {}'.format(path_inference_graph))

path_restore_model = 'logs/2/tfmodel-84' #'logs/2/tfmodel-5' or None
logger.info('Path to restored (already done some training) model\n' +
            '    (if available): {}'.format(path_restore_model))


# In[3]:


################################################################################
################################################################################

# hyper-parameters:
m_height = 13  # multiplicate of image height size -> network is designed so 
    # that it can take images with shape of multiples of m
m_width = m_height+19  # multiplicate of image width size -> network 
    # is designed so that it can take images with shape of multiples of m
IMAGE_SHAPE = [32 * m_height, 32 * m_width, 3]  # complete image size 
    # [436, 1024, 3] # Narihira2015 use [M*32=13*32=416, 416, 3]
logger.info('Trained on images of shape: {}'.format(IMAGE_SHAPE))

INITIAL_LEARNING_RATE = 5e-4
logger.info('Initial learning rate: {}'.format(INITIAL_LEARNING_RATE))

LOSS_TYPE = 'berhu'  # or 'mse'
logger.info('Type of loss function: {}'.format(LOSS_TYPE))

LOSS_LAMBDA = 0.5  # or 0 or 1
logger.info('Loss regularizer lambda value: {}'.format(LOSS_LAMBDA))

# probability that a neuron's output is kept during dropout (only during 
# training!!!, testing/validation -> 1.0):
# DROPOUT_RATE = 0.5
BATCH_SIZE = 8  # nr of data which is put through the network before updating 
    # it, as default use: 32. 
    # BATCH_SIZE determines how many data samples are loaded in the memory (be 
    # careful with memory space)
logger.info('Batch size: {}'.format(BATCH_SIZE))

NUM_EPOCHS = 2  # nr of times the training process loops through the 
    # complete training data set (how often is the tr set 'seen')
    # if you have 1000 training examples, and your batch size is 500, then it
    # will take 2 iterations to complete 1 epoch.
logger.info('# Epochs: {}'.format(NUM_EPOCHS))

DISPLAY_STEP = 2  # every DIPLAY_STEP'th training iteration information is 
    # printed (default: 100)
logger.info('Report training set loss every {} iteration.'.format(DISPLAY_STEP))

SAVE_STEP = 2  # every SAVE_STEP'th training iteration a summary file is 
    # written to LOGS_PATH and checkpoint files are saved
logger.info('Write summary and checkpoints to file every ' +
            '{}-th iteration.'.format(SAVE_STEP))
DEVICE = '/cpu:0'  # device on which the variable is saved/processed
logger.info('Device setting: {}'.format(DEVICE))

################################################################################
################################################################################


# In[4]:


# load meta graph (inference graph)
# how to work with restored models:
# https://www.tensorflow.org/programmers_guide/meta_graph
# http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
saver_restore = tf.train.import_meta_graph(path_inference_graph, 
                                           clear_devices=True)
logger.debug('Restored inference graph.')


# In[5]:


# save default graph in variable:
graph = tf.get_default_graph()


# In[6]:


# plot imported inference graph:
plt_help.show_graph(graph.as_graph_def())


# In[7]:


# lets get the input
x = graph.get_tensor_by_name(name='input:0')

# setup target output classes (ground truth):
y_albedo_label = tf.placeholder(dtype=tf.float32, 
                                shape=[None] + IMAGE_SHAPE, 
                                name='out_albedo')
y_shading_label = tf.placeholder(dtype=tf.float32, 
                                 shape=[None] + IMAGE_SHAPE, 
                                 name='out_shading')

# bool variable that indicates if we are in training mode (training=True) or 
# valid/test mode (training=False) this indicator is important if dropout or/and
# batch normalization is used.
training = graph.get_tensor_by_name(name='is_training:0')

invalid_px_mask = tf.placeholder(dtype=tf.float32, 
                                 shape=[None] + IMAGE_SHAPE, 
                                 name='invalid_px_mask')

# get graph output nodes:
y_albedo_pred = graph.get_tensor_by_name(name='deconv_s2out_albedo/BiasAdd:0')
y_shading_pred = graph.get_tensor_by_name(name='deconv_s2out_shading/BiasAdd:0')
# y_albedo_pred = tf.clip_by_value(t=y_albedo, clip_value_min=0, 
#                                  clip_value_max=1, name='0_1_clipping_albedo')
# y_shading_pred = tf.clip_by_value(t=y_shading, clip_value_min=0,
#                                   clip_value_max=1, 
#                                   name='0_1_clipping_shading')


# In[8]:


valid_mask = cnnhelp.get_valid_pixels(image=x, invalid_mask=invalid_px_mask)
loss = cnnhelp.loss_fct(label_albedo=y_albedo_label,
                        label_shading=y_shading_label,
                        prediction_albedo=y_albedo_pred, 
                        prediction_shading=y_shading_pred,
                        lambda_=LOSS_LAMBDA,
                        loss_type=LOSS_TYPE, 
                        valid_mask=valid_mask, 
                        log=True)

# loss = cnnhelp.loss_fct(label_albedo=y_albedo_label, 
#                         label_shading=y_shading_label, 
#                         prediction_albedo=y_albedo_pred, 
#                         prediction_shading=y_shading_pred, 
#                         lambda_=0.5)
logger.debug('Defined loss.')

# Use an AdamOptimizer to train the network:
with tf.name_scope('optimization'):
    opt_step = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE).minimize(loss)
logger.debug('Defined optimization method.')


# In[9]:


# to get every summary defined above we merge them to get one target:
merge_train_summaries = tf.summary.merge_all()

# define a FileWriter op which writes summaries defined above to disk:
summary_writer = tf.summary.FileWriter(LOGS_PATH)

# Create a saver for writing training checkpoints.
saver = tf.train.Saver(max_to_keep=NUM_EPOCHS)


# In[10]:


# introduce some validation set specific summaries
# These summaries need to be defined blow the function 
# merge_train_summaries = tf.summary.merge_all()
# because the validation set summaries are added to the summary writer at 
# different times. If they had been summarized with the training summaries they 
# would have to be defined at times where merge_train_summaries are added to the
# summary writer
with tf.name_scope('loss/'):
    valid_loss = tf.placeholder(dtype=tf.float32)
    valid_loss_summary = tf.summary.scalar(name='validation_loss', 
                                           tensor=valid_loss)
logger.debug('Defined validation loss')


# In[11]:


# # plot complete graph:
# plt_help.show_graph(graph.as_graph_def())
logger.debug('Finished building training graph.')
logger.info('Total parameters of network: {}'.format(nnhelp.network_params()))


# In[ ]:


# import data:
# import training data:
file = 'sample_data_sintel_shading_train.csv'
df_train = pd.read_csv(DATA_DIR + file, sep=',', header=None,
                       names=['img', 'alb', 'shad', 'invalid'])
# compolete image paths:
df_train = DATA_DIR + df_train
# enable this line to train on only one image:
df_train1 = df_train.loc[[0]]
# replicate this row 100 times:
df_train = pd.concat([df_train1]*100).reset_index(drop=True)
# instantiate a data queue for feeding data in (mini) batches to cnn:
data_train = iq.DataQueue(df=df_train, batch_size=BATCH_SIZE,
                          num_epochs=NUM_EPOCHS)
logger.debug('Imported training data from\n    {}'.format(DATA_DIR + file))

# import validation data set: 
# why not using the whole validation set for validation at once? 
# - limited memory space.
#  -> After each training epoch we will use the complete validation dataset
#     to calculate the error/accuracy on the validation set
file = 'sample_data_sintel_shading_valid.csv'
df_valid = pd.read_csv(DATA_DIR + file, sep=',', header=None,
                       names=['img', 'alb', 'shad', 'invalid'])
# compolete image paths:
df_valid = DATA_DIR + df_valid
# instantiate a data queue for feeding data in (mini) batches to cnn:
data_valid = iq.DataQueue(df=df_valid, batch_size=BATCH_SIZE,
                          num_epochs=NUM_EPOCHS)
logger.debug('Imported validation data from\n    {}'.format(DATA_DIR + file))


# In[ ]:


################################################################################
logger.info('Start training:')
# Initialization:
# Op that initializes global variables in the graph:
init_global = tf.global_variables_initializer()
# Op that initializes local variables in the graph:
init_local = tf.local_variables_initializer()

# config = tf.ConfigProto(allow_soft_placement = True,
#                         intra_op_parallelism_threads=3,
#                         log_device_placement=False)
config = tf.ConfigProto(device_count = {'GPU': 1},
                        intra_op_parallelism_threads=3)
with tf.Session(config=config) as sess: 
# with tf.Session() as sess:
    ############################################################################
    # initialize all variables:
    sess.run([init_global, init_local])
    
    if path_restore_model:
        # restore saved model parameters (weights, biases, etc):
        saver_restore.restore(sess, path_restore_model)
    
    # Adds a Graph to the event file.
    # create summary that give output (TensorFlow op that output protocol 
    # buffers containing 'summarized' data) of the built Tensorflow graph:
    summary_writer.add_graph(sess.graph)
    
    # start timer for total training time:
    start_total_time = time.time()
    # set timer to measure the displayed training steps:
    start_time = start_total_time
    
    ############################################################################
    # Training:
    # train loop
    # train until all data is processed (queue empty),
    # number of iterations depends on number of data, number of epochs and 
    # batch size:
    iter_start = data_train.iter_left
    logger.info('For training it takes {}\n'.format(iter_start) +
                '    (= # data / batch_size * epochs) iterations to loop ' +
                'through {} samples of\n    '.format(data_train.df.shape[0]) +
                'training data over {} '.format(data_train.num_epochs) +
                'epochs summarized in batches of size ' + 
                '{}.\n'.format(data_train.batch_size) +
                '    So, there are # data / batch_size = ' +
                '{} '.format(int(data_train.df.shape[0]/data_train.batch_size))+
                'iterations per epoch.')
    
    while data_train.iter_left >= 0:
        try:
            # take a (mini) batch of the training data:
            deq_train = data_train.dequeue()
            img_b, alb_b, shad_b, inv_b = iq.next_batch(deq=deq_train, 
                                                        output_shape=IMAGE_SHAPE,
                                                        is_scale=True,
                                                        is_flip=True, 
                                                        is_rotated=True,
                                                        norm=True)
            # run training/optimization step:
            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.

            feed_dict = {x: img_b,
                         y_albedo_label: alb_b,
                         y_shading_label: shad_b,
                         invalid_px_mask: inv_b,
                         training: True}
            sess.run(opt_step, feed_dict=feed_dict)

            # report training set accuracy every DISPLAY_STEP-th step:
            if (data_train.num_iter) % DISPLAY_STEP == 0:
                # console output:
                feed_dict = {x: img_b,
                             y_albedo_label: alb_b,
                             y_shading_label: shad_b,
                             invalid_px_mask: inv_b,
                             training: False}
                train_loss = sess.run(loss, 
                                      feed_dict=feed_dict)
                dur_time = time.time() - start_time
                dur_time = ghelp.get_time_format(time_in_sec=dur_time)
                dur_time = ghelp.time_tuple_to_str(time_tuple=dur_time)
                logger.info('iteration {}: '.format(data_train.num_iter) +
                            'training loss ' + 
                            '{tr_loss:.2f} (ET: '.format(tr_loss=train_loss) +
                            '{}).'.format(dur_time))
                # reset timer to measure the displayed training steps:
                start_time = time.time()

        ########################################################################
            # Validation:
            # display validation set accuracy and loss after each completed 
            # epoch (1 epoch ^= data_train.df.shape[0] / data_train.batch_size 
            # training steps => steps per epoch)
            val_epoch = int(data_train.df.shape[0] / data_train.batch_size)
            if data_train.num_iter % val_epoch == 0:
                # After each training epoch we will use the complete validation 
                # data set to calculate the error/accuracy on the validation 
                # set:
                # loop through one validation data set epoch:
                validation_loss = 0
                valid_steps_per_epoch = int(data_valid.df.shape[0] / 
                                            data_valid.batch_size)
                for j in range(valid_steps_per_epoch):
                    # DISCLAIMER: we do not run the opt_step here (on 
                    # the validation data set) because we do not want to train
                    # our network on the validation set. Important for batch 
                    # normalization and dropout (training -> False).
                    # get validation data set (mini) batch:
                    lst = iq.next_batch(deq=data_valid.dequeue(), 
                                        output_shape=IMAGE_SHAPE,
                                        is_scale=False,
                                        is_flip=False,
                                        is_rotated=False,
                                        norm=True)
                    img_b_val, alb_b_val, shad_b_val, inv_b_val = lst
                    
                    # calculate the mean loss of this validation batch and sum 
                    # it with the previous mean batch losses:
                    feed_dict = {x: img_b_val,
                                 y_albedo_label: alb_b_val,
                                 y_shading_label: shad_b_val,
                                 invalid_px_mask: inv_b_val,
                                 training: False}
                    validation_loss += sess.run(loss, 
                                                feed_dict=feed_dict)

                # adding a mean loss summary op (for tensorboard). 
                # we need to divide the accumulated loss from above by the 
                # iteration steps (steps_per_epoch):
                feed_dict = {valid_loss: validation_loss/valid_steps_per_epoch}
                validation_loss_total = sess.run(valid_loss_summary, 
                                                 feed_dict=feed_dict)
                summary_writer.add_summary(summary=validation_loss_total, 
                                           global_step=data_train.num_iter)
                dur_time = time.time() - start_time
                dur_time = ghelp.get_time_format(time_in_sec=dur_time)
                dur_time = ghelp.time_tuple_to_str(time_tuple=dur_time)
                logger.info('Validation scores after epoch ' + 
                            '{} '.format(data_train.completed_epochs + 1) + 
                            '(iteration {}):\n'.format(data_train.num_iter) +
                            '    # validation data: ' +
                            '{}, mean loss: '.format(data_valid.df.shape[0]) +
                            '{:.2f}'.format(validation_loss/valid_steps_per_epoch)+
                            ' (ET: {}).'.format(dur_time))
                # reset timer to measure the displayed training steps:
                start_time = time.time()

            if data_train.num_iter % SAVE_STEP == 0:
                # save checkpoint files to disk:
                save_path = saver.save(sess, LOGS_PATH + 'tfmodel',
                                       global_step=data_train.num_iter)
                feed_dict = {x: img_b,  
                             y_albedo_label: alb_b,
                             y_shading_label: shad_b,
                             invalid_px_mask: inv_b,
                             training: False}
                s = sess.run(merge_train_summaries, feed_dict=feed_dict)
                # adds a Summary protocol buffer to the event file 
                #     (global_step: Number. Optional global step value to record
                #     with the summary. Each stepp i is assigned to the 
                #     corresponding summary parameter.)
                summary_writer.add_summary(summary=s, 
                                           global_step=data_train.num_iter)
                dur_time = time.time() - start_time
                dur_time = ghelp.get_time_format(time_in_sec=dur_time)
                dur_time = ghelp.time_tuple_to_str(time_tuple=dur_time)
                logger.info('Saved data (iteration ' + 
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


# In[ ]:


# !tensorboard --logdir /logs/1

