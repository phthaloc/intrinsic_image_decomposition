
# coding: utf-8

# # Import inference graph and extend it to a training graph:
# 
# To extend the graph, we have to add:
#     - input data structure
#     - loss function
#     - optimization op
#     
# To build the graph it is necessary to know the node names of all relevant layers, placeholders etc. for restoring the model later.
# 
# pay ATTENTION to:
#     - the imported model must be in the same graph as the nodes which are added later
#         -> load the model first to the default graph, then add further ops
#     
# also test to input input-images of different sizes (multiples of 32px). 
# It might not work because input placeholder is defined fix.
# perhaps input node should not be saved inside the model?!
# 
# output of model has name (deconv_s2out_shading/BiasAdd:0 and deconv_s2out_albedo/BiasAdd:0). 
# these are to complicated?!
# 
# 
# To plot all graphs directly in this notebook, run jupyter form terminal like this:
#     jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000000

# In[1]:


import sys
sys.path.append('./util')
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import input_queues as iq
import cnn_model
import plot_helpers as plt_help
import general_helpers as ghelp
import cnn_helpers as cnnhelp

print('Python version: \n' + sys.version)
print('Tensorflow version: \n' + tf.__version__)

# data path constants:
# DATA_DIR = '../data/mnist/'
DATA_DIR = 'data/'
PREDICT_PATH = ''
path_inference_graph = ['logs/inference_graphs/narihira2015/' +
                        'tfmodel_inference.meta']
path_inference_graph = path_inference_graph[0]
path_restore_model = None #'logs/2/tfmodel-5'
LOGS_PATH = 'logs/1/'  # path to summary files

# hyper-parameters:
m_height = 13  # multiplicate of image height size -> network is designed so 
    # that it can take images with shape of multiples of m
m_width = m_height  # multiplicate of image width size -> network 
    # is designed so that it can take images with shape of multiples of m
IMAGE_SHAPE = [32 * m_height, 32 * m_width, 3]  # complete image size 
    # [436, 1024, 3] # Narihira2015 use [M*32=13*32=416, 416, 3]
INITIAL_LEARNING_RATE = 1e-5
# probability that a neuron's output is kept during dropout (only during 
# training!!!, testing/validation -> 1.0):
# DROPOUT_RATE = 0.5
BATCH_SIZE = 16  # nr of data which is put through the network before updating 
    # it, as default use: 32. 
# BATCH_SIZE determines how many data samples are loaded in the memory (be 
# careful with memory space)
NUM_EPOCHS = 16  # nr of times the training process loops through the 
    # complete training data set (how often is the tr set 'seen')
    # if you have 1000 training examples, and your batch size is 500, then it
    # will take 2 iterations to complete 1 epoch.

DISPLAY_STEP = 100  # every DIPLAY_STEP'th training iteration information is 
    # printed (default: 100)
SUMMARY_STEP = 32  # every SUMMARY_STEP'th training iteration a summary file is 
    # written to LOGS_PATH
DEVICE = '/gpu:0'  # device on which the variable is saved/processed


# In[2]:


# load meta graph (inference graph)
# how to work with restored models:
# https://www.tensorflow.org/programmers_guide/meta_graph
# http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
saver_restore = tf.train.import_meta_graph(path_inference_graph, 
                                           clear_devices=True)


# In[3]:


# save default graph in variable:
graph = tf.get_default_graph()


# In[4]:




# In[5]:


with tf.name_scope('data'):
    # import training data set
    file = 'data_sintel_shading_train.csv'
    data_train = iq.SintelDataInputQueue(path_csv_file = DATA_DIR + file,
                                         batch_size=BATCH_SIZE, 
                                         num_epochs=NUM_EPOCHS, 
                                         nr_data=None)
    # if data_augmentation=True: images are randomly rotated in range (-15, 15) 
    # deg and randomly horizontally flipped:
    data_train_out = data_train.next_batch(image_shape=IMAGE_SHAPE, 
                                data_augmentation=True)
    _, _, _, imgs_batch_tr, albedo_batch_tr, shading_batch_tr = data_train_out

    # import validation data set: 
    # why not using the whole validation set for validation at once? 
    # - limited memory space.
    #  -> After each training epoch we will use the complete validation dataset
    #     to calculate the error/accuracy on the validation set
    file = 'data_sintel_shading_valid.csv'
    data_valid = iq.SintelDataInputQueue(path_csv_file = DATA_DIR + file,
                                         batch_size=5, 
                                         num_epochs=NUM_EPOCHS,
                                         nr_data=None)
    # if data_augmentation=True: images are randomly rotated in range (-15, 15) 
    # deg and randomly horizontally flipped:
    data_val_out = data_valid.next_batch(image_shape=IMAGE_SHAPE,
                                         data_augmentation=False)
    _, _, _, imgs_batch_val, albedo_batch_val, shading_batch_val = data_val_out

    
#     # testing data set: 
#     file = 'data_sintel_shading_test.csv'
#     data_test = iq.SintelDataInputQueue(path_csv_file = DATA_DIR + file,
#                                         batch_size=1, 
#                                         num_epochs=NUM_EPOCHS, 
#                                         nr_data=None)
#     # if data_augmentation=True: images are randomly rotated in range (-15, 15)
#     # deg and randomly horizontally flipped:
#     data_te_out = data_test.next_batch(image_shape=IMAGE_SHAPE,
#                                        data_augmentation=False)
#     image_path_batch_test, albedo_path_batch_test, shading_path_batch_test, \
#         images_batch_test, albedo_batch_test, shading_batch_test = data_te_out
    
#     # for the test set create also 
#     image_path_test, albedo_label_path_test, shading_label_path_test = data_test.read_csv_file(record_defaults=[[''], [''], ['']])
    
#     images_test = data_test.read_image(image_path=image_path_test)
#     labels_albedo_test = data_test.read_image(image_path=albedo_label_path_test)
#     labels_shading_test = data_test.read_image(image_path=shading_label_path_test)
#     images_test, labels_albedo_test, labels_shading_test = data_test.random_crop_images_and_labels(image_and_labels=[images_test,
#                                                                                                                      labels_albedo_test,
#                                                                                                                      labels_shading_test],
#                                                                                                    channels=[IMAGE_SHAPE[-1]]*3,
#                                                                                                    spatial_shape=IMAGE_SHAPE[:2],
#                                                                                                    data_augmentation=False)


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

# Use an AdamOptimizer to train the network:
with tf.name_scope('optimization'):
    optimization_step = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE).minimize(loss)


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


# In[10]:




# In[11]:


with tf.Session() as sess:
    
    ############################################################################
    # Initialization:
    # Op that initializes global variables in the graph:
    init_global = tf.global_variables_initializer()
    # Op that initializes local variables in the graph:
    init_local = tf.local_variables_initializer()
    # initialize all variables:
    sess.run([init_global, init_local])
    if path_restore_model:
        # restore saved model parameters (weights, biases, etc):
        saver_restore.restore(sess, path_restore_model)
    # Adds a Graph to the event file.
    #     create summary that give output (TensorFlow op that output protocol 
    #     buffers containing 'summarized' data) of the built Tensorflow graph:
    summary_writer.add_graph(sess.graph)
    
    # start timer for total training time:
    start_total_time = time.time()
    # set timer to measure the displayed training steps:
    start_time = start_total_time
    
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    ############################################################################
    # Training:
    count_epoch = 0
    # train loop
    #     train for until all data is used
    #     number of iterations depends on number of data, number of epochs and 
    #     batch size:
    train_iters = int(data_train.nr_data / data_train.batch_size * 
                      data_train.num_epochs)

    print('INFO: For training it takes {} '.format(train_iters) +
          '(= # data / batch_size * epochs) iterations to loop through ' +
          '{} samples of training data over '.format(data_train.nr_data) +
          '{} epochs summarized in batches '.format(data_train.num_epochs) +
          'of size {}.\n'.format(data_train.batch_size) +
          'So, there are # data / batch_size = ' +
          '{} '.format(int(data_train.nr_data / data_train.batch_size)) + 
          'iterations per epoch.\n')
    
    for i in range(train_iters):
        # take a (mini) batch of the training data:
        # method of the DataSet class
        lst = [imgs_batch_tr, albedo_batch_tr, shading_batch_tr]
        next_imgs_batch, next_albedo_batch, next_shad_batch = sess.run(lst)

        # run training/optimization step:
        #     Run one step of the model.  The return values are the activations
        #     from the `train_op` (which is discarded) and the `loss` Op.  To
        #     inspect the values of your Ops or variables, you may include them
        #     in the list passed to sess.run() and the value tensors will be
        #     returned in the tuple from the call.

        feed_dict = {x: next_imgs_batch,
                     y_albedo_label: next_albedo_batch,
                     y_shading_label: next_shad_batch,
                     training: True}
        sess.run(optimization_step, feed_dict=feed_dict)

        # report training set accuracy every DISPLAY_STEP-th step:
        if i % DISPLAY_STEP == 0:
            # console output:
            feed_dict = {x: next_imgs_batch,
                         y_albedo_label: next_albedo_batch,
                         y_shading_label: next_shad_batch,
                         training: False}
            train_loss = sess.run(loss, 
                                  feed_dict=feed_dict)
            duration_time = time.time() - start_time
            duration_time = ghelp.get_time_format(time_in_sec=duration_time)
            print('iteration {iteration}: training loss '.format(iteration=i) + 
                  '{tr_loss:.2f} (ET: '.format(tr_loss=train_loss) +
                  '{dur_time_h:02}:'.format(dur_time_h=duration_time[0]) +
                  '{dur_time_min:02}:'.format(dur_time_min=duration_time[1]) + 
                  '{dur_time_s:02} h).'.format(dur_time_s=duration_time[2]))
            # reset timer to measure the displayed training steps:
            start_time = time.time()

        ########################################################################
        # Validation:
        # display validation set accuracy and loss after each completed epoch 
        # (1 epoch ^= data.nr_data / data.batch_size training steps
        # => steps per epoch)
        # the term (i + 1) in line below comes from: iteration starts at 0 not 
        # at 1
        if (((i + 1) % int(data_train.nr_data / data_train.batch_size) == 0) and
            (i != 0)):
            count_epoch += 1
            # save checkpoint files to disk:
            save_path = saver.save(sess, LOGS_PATH + 'tfmodel', global_step=i)
            print('Validation scores after epoch {} '.format(count_epoch) + 
                  '(step {}):\n'.format(i) +
                  '    Model saved in file: {}.'.format(save_path))
            # After each training epoch we will use the complete validation data
            #     set to calculate the error/accuracy on the validation set:
            # loop through one validation data set epoch:
            validation_loss = 0
            valid_steps_per_epoch = int(data_valid.nr_data / 
                                        data_valid.batch_size)
            for j in range(valid_steps_per_epoch):
                # DISCLAIMER: we do not run the optimization_step here (on the 
                #     validation data set) because we do not want to train our 
                #     network on the validation set. Important for batch 
                #     normalization and dropout.
                # get validation data set (mini) batch:
                lst = [imgs_batch_val, albedo_batch_val, shading_batch_val]
                next_imgs_batch_val, next_albedo_batch_val, next_shad_batch_val = sess.run(lst)

                # calculate the mean loss of this validation batch and sum it 
                # with the previous mean batch losses:
                feed_dict = {x: next_imgs_batch_val,
                             y_albedo_label: next_albedo_batch_val,
                             y_shading_label: next_shad_batch_val,
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
                                       global_step=i)
            print('    Num validation data: {}, '.format(data_valid.nr_data) + 
                  'mean loss: ' +
                  '{ml:.2f}'.format(ml=validation_loss / valid_steps_per_epoch))
            
        if i % SUMMARY_STEP == 0:
            feed_dict = {x: next_imgs_batch,  
                         y_albedo_label: next_albedo_batch,
                         y_shading_label: next_shad_batch, 
                         training: False}
            s = sess.run(merge_train_summaries, feed_dict=feed_dict)
            # adds a Summary protocol buffer to the event file 
            #     (global_step: Number. Optional global step value to record 
            #     with the summary. Each stepp i is assigned to the 
            #     corresponding summary parameter.)
            summary_writer.add_summary(summary=s, global_step=i)
        
    
    end_total_time = time.time() - start_total_time
    end_total_time = ghelp.get_time_format(end_total_time)
    print('\nTraining done... total training time: ' + 
          '{h:02}:'.format(h=end_total_time[0]) +
          '{m:02}:{s:02} h.'.format(m=end_total_time[1], s=end_total_time[2]))
    
    ############################################################################
    
    coord.request_stop()
    coord.join(threads)


# In[12]:


# print all op and tensor names in default graph:
# len([print(n.name) for n in graph.as_graph_def().node])
# list all global variables:
# tf.global_variables()


# In[ ]:





# In[13]:


# !tensorboard --logdir ./logs/2


# In[ ]:




