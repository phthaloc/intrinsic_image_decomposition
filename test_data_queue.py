
# coding: utf-8

# # Test data queue:

# In[1]:


import sys
sys.path.append('./util')
import time
import numpy as np
import pandas as pd
import tensorflow as tf

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
# path_inference_graph = ['/Users/udodehm/Downloads/camp_depth_irolaina/ResNet_pretrained/ResNet-L50.meta']
# path_inference_graph = ['vgg16/vgg16.tfmodel']

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
BATCH_SIZE = 8  # nr of data which is put through the network before updating 
    # it, as default use: 32. 
# BATCH_SIZE determines how many data samples are loaded in the memory (be 
# careful with memory space)
NUM_EPOCHS = 3  # nr of times the training process loops through the 
    # complete training data set (how often is the tr set 'seen')
    # if you have 1000 training examples, and your batch size is 500, then it
    # will take 2 iterations to complete 1 epoch.

DISPLAY_STEP = 2  # every DIPLAY_STEP'th training iteration information is 
    # printed (default: 100)
SUMMARY_STEP = 2  # every SUMMARY_STEP'th training iteration a summary file is 
    # written to LOGS_PATH
DEVICE = '/cpu:0'  # device on which the variable is saved/processed


# In[12]:


with tf.name_scope('data'):
    # import training data set
    file = 'sample_data_sintel_shading_train.csv'
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
    file = 'sample_data_sintel_shading_valid.csv'
    data_valid = iq.SintelDataInputQueue(path_csv_file = DATA_DIR + file,
                                         batch_size=5, 
                                         num_epochs=NUM_EPOCHS,
                                         nr_data=None)
    # if data_augmentation=True: images are randomly rotated in range (-15, 15) 
    # deg and randomly horizontally flipped:
    data_val_out = data_valid.next_batch(image_shape=IMAGE_SHAPE,
                                         data_augmentation=False)
    _, _, _, imgs_batch_val, albedo_batch_val, shading_batch_val = data_val_out

    
    # testing data set: 
    file = 'sample_data_sintel_shading_test.csv'
    data_test = iq.SintelDataInputQueue(path_csv_file = DATA_DIR + file,
                                        batch_size=16, 
                                        num_epochs=NUM_EPOCHS, 
                                        nr_data=None)
    # if data_augmentation=True: images are randomly rotated in range (-15, 15)
    # deg and randomly horizontally flipped:
    data_te_out = data_test.next_batch(image_shape=IMAGE_SHAPE,
                                       data_augmentation=False)
    image_path_batch_test, albedo_path_batch_test, shading_path_batch_test, images_batch_test, albedo_batch_test, shading_batch_test = data_te_out
    
    # for the test set create also 
    image_path_test, albedo_label_path_test, shading_label_path_test = data_test.read_csv_file(record_defaults=[[''], [''], ['']])
    
    images_test = data_test.read_image(image_path=image_path_test)
    labels_albedo_test = data_test.read_image(image_path=albedo_label_path_test)
    labels_shading_test = data_test.read_image(image_path=shading_label_path_test)
    images_test, labels_albedo_test, labels_shading_test = data_test.random_crop_images_and_labels(image_and_labels=[images_test,
                                                                                                                     labels_albedo_test,
                                                                                                                     labels_shading_test],
                                                                                                   channels=[IMAGE_SHAPE[-1]]*3,
                                                                                                   spatial_shape=IMAGE_SHAPE[:2],
                                                                                                   data_augmentation=False)


# In[ ]:


# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
with tf.Session() as sess:
    
    ############################################################################
    # Initialization:
    # Op that initializes global variables in the graph:
    init_global = tf.global_variables_initializer()
    # Op that initializes local variables in the graph:
    init_local = tf.local_variables_initializer()
    # initialize all variables:
    sess.run([init_global, init_local])
    # start timer for total training time:
    start_total_time = time.time()
    # set timer to measure the displayed training steps:
    start_time = start_total_time
    
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    
    iters = int(data_test.nr_data / data_test.batch_size * 
                      data_test.num_epochs)
    print('available iterations: ' + iters)
    iters_used = int(NUM_EPOCHS * 0.75)
    print('used iterations: ' + iters_used)
    
    for i in range(iters_used):
        nxt_img_path_batch_te, nxt_albedo_path_batch_te, \
            nxt_shad_path_batch_te, nxt_img_batch_te, nxt_albedo_batch_te, \
            nxt_shad_batch_te = sess.run([image_path_batch_test,
                                          albedo_path_batch_test, 
                                          shading_path_batch_test,
                                          images_batch_test, 
                                          albedo_batch_test,
                                          shading_batch_test])
        print(nxt_img_path_batch_te)
#     ############################################################################
#     # Training:
#     count_epoch = 0
#     # train loop
#     #     train for until all data is used
#     #     number of iterations depends on number of data, number of epochs and 
#     #     batch size:
#     train_iters = int(data_train.nr_data / data_train.batch_size * 
#                       data_train.num_epochs)

#     print('INFO: For training it takes {} '.format(train_iters) +
#           '(= # data / batch_size * epochs) iterations to loop through ' +
#           '{} samples of training data over '.format(data_train.nr_data) +
#           '{} epochs summarized in batches '.format(data_train.num_epochs) +
#           'of size {}.\n'.format(data_train.batch_size) +
#           'So, there are # data / batch_size = ' +
#           '{} '.format(int(data_train.nr_data / data_train.batch_size)) + 
#           'iterations per epoch.\n')
    
#     for i in range(train_iters):
#         # take a (mini) batch of the training data:
#         # method of the DataSet class
#         lst = [imgs_batch_tr, albedo_batch_tr, shading_batch_tr]
#         next_imgs_batch, next_albedo_batch, next_shad_batch = sess.run(lst)
        
    
#     end_total_time = time.time() - start_total_time
#     end_total_time = ghelp.get_time_format(end_total_time)
#     print('\nTraining done... total training time: ' + 
#           '{h:02}:'.format(h=end_total_time[0]) +
#           '{m:02}:{s:02} h.'.format(m=end_total_time[1], s=end_total_time[2]))
    
    ############################################################################
    
    coord.request_stop()
    coord.join(threads)



