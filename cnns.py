
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

LOGS_PATH = 'logs/1/'  # path to summary files
# create logger (write to file and stdout):
logger = ghelp.create_logger(filename=LOGS_PATH + 'training.log')

logger.debug('Python version: \n' + sys.version + 
             '\n    Tensorflow version: \n' + tf.__version__)

# data path constants:
# DATA_DIR = '../data/mnist/'
DATA_DIR = 'data/'
# DATA_DIR = '/usr/udo/data/'
PREDICT_PATH = ''
path_inference_graph = ['logs/inference_graphs/narihira2015/' +
                        'tfmodel_inference.meta']
# path_inference_graph = ['/Users/udodehm/Downloads/camp_depth_irolaina/' + '
#                         'ResNet_pretrained/ResNet-L50.meta']
# path_inference_graph = ['vgg16/vgg16.tfmodel']
path_inference_graph = path_inference_graph[0]
logger.info('Path to infrence graph:\n    {}'.format(path_inference_graph))

path_restore_model = None #'logs/2/tfmodel-5'
logger.info('Path to restored (already done some training) model\n' +
            '    (if available): {}'.format(path_restore_model))

# hyper-parameters:
m_height = 13  # multiplicate of image height size -> network is designed so 
    # that it can take images with shape of multiples of m
m_width = m_height  # multiplicate of image width size -> network 
    # is designed so that it can take images with shape of multiples of m
IMAGE_SHAPE = [32 * m_height, 32 * m_width, 3]  # complete image size 
    # [436, 1024, 3] # Narihira2015 use [M*32=13*32=416, 416, 3]
logger.info('Trained on images of shape: {}'.format(IMAGE_SHAPE))

INITIAL_LEARNING_RATE = 1e-5
logger.info('Initial learning rate: {}'.format(INITIAL_LEARNING_RATE))

# probability that a neuron's output is kept during dropout (only during 
# training!!!, testing/validation -> 1.0):
# DROPOUT_RATE = 0.5
BATCH_SIZE = 8  # nr of data which is put through the network before updating 
    # it, as default use: 32. 
    # BATCH_SIZE determines how many data samples are loaded in the memory (be 
    # careful with memory space)
    
NUM_EPOCHS = 2  # nr of times the training process loops through the 
    # complete training data set (how often is the tr set 'seen')
    # if you have 1000 training examples, and your batch size is 500, then it
    # will take 2 iterations to complete 1 epoch.

DISPLAY_STEP = 2  # every DIPLAY_STEP'th training iteration information is 
    # printed (default: 100)
SUMMARY_STEP = 2  # every SUMMARY_STEP'th training iteration a summary file is 
    # written to LOGS_PATH
DEVICE = '/gpu:0'  # device on which the variable is saved/processed


# In[2]:


# graph = tf.Graph()
  
# # Set the new graph as the default.
# with graph.as_default():
  
#     # Open the graph-def file for binary reading.
#     path = path_inference_graph
#     with tf.gfile.FastGFile(path, 'rb') as file:
#         # The graph-def is a saved copy of a TensorFlow graph.
#         # First we need to create an empty graph-def.
#         graph_def = tf.GraphDef()

#         # Then we load the proto-buf file into the graph-def.
#         graph_def.ParseFromString(file.read())

#         # Finally we import the graph-def to the default TensorFlow graph.
#         tf.import_graph_def(graph_def, name='')


# In[3]:


# load meta graph (inference graph)
# how to work with restored models:
# https://www.tensorflow.org/programmers_guide/meta_graph
# http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
saver_restore = tf.train.import_meta_graph(path_inference_graph, 
                                           clear_devices=True)
logger.debug('Restored inference graph.')


# In[4]:


# save default graph in variable:
graph = tf.get_default_graph()


# In[5]:


# # plot imported inference graph:
# plt_help.show_graph(graph.as_graph_def())


# In[6]:


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
# get graph output nodes:
y_albedo_pred = graph.get_tensor_by_name(name='deconv_s2out_albedo/BiasAdd:0')
y_shading_pred = graph.get_tensor_by_name(name='deconv_s2out_shading/BiasAdd:0')
# y_albedo_pred = tf.clip_by_value(t=y_albedo, clip_value_min=0, 
#                                  clip_value_max=1, name='0_1_clipping_albedo')
# y_shading_pred = tf.clip_by_value(t=y_shading, clip_value_min=0,
#                                   clip_value_max=1, 
#                                   name='0_1_clipping_shading')


# In[7]:


loss = cnnhelp.loss_fct(label_albedo=y_albedo_label, 
                        label_shading=y_shading_label, 
                        prediction_albedo=y_albedo_pred, 
                        prediction_shading=y_shading_pred, 
                        lambda_=0.5)
logger.debug('Defined loss.')

# Use an AdamOptimizer to train the network:
with tf.name_scope('optimization'):
    opt_step = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE).minimize(loss)
logger.debug('Defined optimization method.')


# In[8]:


# to get every summary defined above we merge them to get one target:
merge_train_summaries = tf.summary.merge_all()

# define a FileWriter op which writes summaries defined above to disk:
summary_writer = tf.summary.FileWriter(LOGS_PATH)

# Create a saver for writing training checkpoints.
saver = tf.train.Saver(max_to_keep=NUM_EPOCHS)


# In[9]:


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


# In[10]:


# # plot complete graph:
# plt_help.show_graph(graph.as_graph_def())
logger.debug('Finished building training graph.')
logger.info('Total parameters of network: {}'.format(nnhelp.network_params()))


# In[12]:


# import data:
# import training data:
file = 'sample_data_sintel_shading_train.csv'
df_train = pd.read_csv(DATA_DIR + file, sep=',', header=None,
                       names=['img', 'alb', 'shad'])
# compolete image paths:
df_train = DATA_DIR + df_train
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
                       names=['img', 'alb', 'shad'])
# compolete image paths:
df_valid = DATA_DIR + df_valid
# instantiate a data queue for feeding data in (mini) batches to cnn:
data_valid = iq.DataQueue(df=df_valid, batch_size=BATCH_SIZE,
                          num_epochs=NUM_EPOCHS)
logger.debug('Imported validation data from\n    {}'.format(DATA_DIR + file))

# testing data set: 
file = 'sample_data_sintel_shading_test.csv'
df_test = pd.read_csv(DATA_DIR + file, sep=',', header=None,
                      names=['img', 'alb', 'shad'])
# compolete image paths:
df_test = DATA_DIR + df_test
# instantiate a data queue for feeding data in (mini) batches to cnn:
data_test = iq.DataQueue(df=df_test, batch_size=BATCH_SIZE,
                         num_epochs=1)
logger.debug('Imported testing data from\n    {}'.format(DATA_DIR + file))

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
    #     train for until all data is used
    #     number of iterations depends on number of data, number of epochs and 
    #     batch size:
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
            img_batch, alb_batch, shad_batch = iq.next_batch(deq=deq_train, 
                                                             shape=IMAGE_SHAPE, 
                                                             is_flip=True, 
                                                             is_rotated=False,
                                                             norm=True)
            # run training/optimization step:
            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.

            feed_dict = {x: img_batch,
                         y_albedo_label: alb_batch,
                         y_shading_label: shad_batch,
                         training: True}
            sess.run(opt_step, feed_dict=feed_dict)

            # report training set accuracy every DISPLAY_STEP-th step:
            if (data_train.num_iter) % DISPLAY_STEP == 0:
                # console output:
                feed_dict = {x: img_batch,
                             y_albedo_label: alb_batch,
                             y_shading_label: shad_batch,
                             training: False}
                train_loss = sess.run(loss, 
                                      feed_dict=feed_dict)
                duration_time = time.time() - start_time
                duration_time = ghelp.get_time_format(time_in_sec=duration_time)
                duration_time = ghelp.time_tuple_to_str(time_tuple=duration_time)
                logger.info('iteration {}: '.format(data_train.num_iter) +
                            'training loss ' + 
                            '{tr_loss:.2f} (ET: '.format(tr_loss=train_loss) +
                            '{}).'.format(duration_time))
                # reset timer to measure the displayed training steps:
                start_time = time.time()

        ########################################################################
            # Validation:
            # display validation set accuracy and loss after each completed 
            # epoch (1 epoch ^= data_train.df.shape[0] / data_train.batch_size 
            # training steps => steps per epoch)
            val_epoch = int(data_train.df.shape[0] / data_train.batch_size)
            if data_train.num_iter % val_epoch == 0:
                # save checkpoint files to disk:
                save_path = saver.save(sess, LOGS_PATH + 'tfmodel',
                                       global_step=data_train.num_iter)

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
                                        shape=IMAGE_SHAPE, 
                                        is_flip=True,
                                        is_rotated=False,
                                        norm=True)
                    img_batch_val, alb_batch_val, shad_batch_val = lst
                    
                    # calculate the mean loss of this validation batch and sum 
                    # it with the previous mean batch losses:
                    feed_dict = {x: img_batch_val,
                                 y_albedo_label: alb_batch_val,
                                 y_shading_label: shad_batch_val,
                                 training: False}
                    validation_loss += sess.run(loss, 
                                                feed_dict=feed_dict)

                # adding a mean loss summary op (for tensorboard). 
                # we need to divide the accumulated loss from above by the 
                # iteration steps (steps_per_epoch):
                feed_dict = {valid_loss: validation_loss / valid_steps_per_epoch}
                validation_loss_total = sess.run(valid_loss_summary, 
                                                 feed_dict=feed_dict)
                summary_writer.add_summary(summary=validation_loss_total, 
                                           global_step=data_train.num_iter)
                duration_time = time.time() - start_time
                duration_time = ghelp.get_time_format(time_in_sec=duration_time)
                duration_time = ghelp.time_tuple_to_str(time_tuple=duration_time)
                logger.info('Validation scores after epoch ' + 
                            '{} '.format(data_train.completed_epochs + 1) + 
                            '(step {}):\n'.format(data_train.num_iter) +
                            '    Model saved in file: {}.\n'.format(save_path) +
                            '    # validation data: ' +
                            '{}, mean loss: '.format(data_valid.df.shape[0]) +
                            '{ml:.2f}'.format(ml=validation_loss / valid_steps_per_epoch) +
                            ' (ET: {}).'.format(duration_time))
                # reset timer to measure the displayed training steps:
                start_time = time.time()

            if data_train.num_iter % SUMMARY_STEP == 0:
                feed_dict = {x: img_batch,  
                             y_albedo_label: alb_batch,
                             y_shading_label: shad_batch, 
                             training: False}
                s = sess.run(merge_train_summaries, feed_dict=feed_dict)
                # adds a Summary protocol buffer to the event file 
                #     (global_step: Number. Optional global step value to record
                #     with the summary. Each stepp i is assigned to the 
                #     corresponding summary parameter.)
                summary_writer.add_summary(summary=s, 
                                           global_step=data_train.num_iter)
        # end while loop when there are no elements left to dequeue:
        except IndexError:
            end_total_time = time.time() - start_total_time
            end_total_time = ghelp.get_time_format(end_total_time)
            end_total_time = ghelp.time_tuple_to_str(time_tuple=end_total_time)
            logger.info('Training done... total training time: ' + 
                        '{}.'.format(end_total_time))
            break
logger.info('Finished training.')


# In[14]:


# !tensorboard --logdir ./logs/1

