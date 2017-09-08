#!/usr/bin/env python

"""
Module that contains queue classes for creating and processing tensorflow
objects (images, labels, etc.)
"""

import os
import numpy as np
import typing
import tensorflow as tf
import abc
import scipy as sp
import scipy.misc
import scipy.ndimage


__author__ = "Udo Dehm"
__copyright__ = "Copyright 2017"
__credits__ = ["Udo Dehm"]
__license__ = ""
__version__ = "0.1"
__maintainer__ = "Udo Dehm"
__email__ = "udo.dehm@mailbox.org"
__status__ = "Development"

__all__ = ['SintelDataInputQueue', 'DataInputQueue', 'next_batch',
           'process_image', 'DataQueue']
 
class DataInputQueue(object):
    def __init__(self,
                 path_csv_file: str,
                 batch_size: int = 2,
                 num_epochs: int = 1, 
                 nr_data: typing.Optional[int] = None) -> None:
        """
        :param path_csv_file: Path to a .csv file that contains list of location
            of data files (like training, validation or testing data) and
            perhaps labels. 
        :param batch_size: number of data (images) that is used for one pass
            (forwards and backwards) through network before parameters
            (weights, biases) are updated
        :param num_epochs: how often each data is used in training step
            (# num_epochs times each image is put through the network)
        :param nr_data: (optional) the number of data. If not given, it is
            automatically calculated from the csv file.
        :type nr_data: int
        """
        
        if not isinstance(batch_size, int) or batch_size < 0:
            raise ValueError('Invalid value, batch_size must be a positive ' +
                             'integer')
        self.batch_size = batch_size
        
        if not isinstance(num_epochs, int) or num_epochs < 0:
            raise ValueError('Invalid value, num_epochs must be a positive ' +
                             'integer')
        self.num_epochs = num_epochs
        
        # define the complete path to the relevant csv data list file
        self.path_csv_file = path_csv_file
        
        self.nr_data = nr_data
        if not self.nr_data:
            self.nr_data = int(sum(1 for line in open(self.path_csv_file)))


    def read_csv_file(self, shuffle=False, record_defaults=[[''], ['']]):
        """
        Builds a file name queue of rows contained in a .csv data list file
        ('*_data_list.csv').  Then it decodes each row and outputs each row in a
        queue. Each row consists of two columns (image path and label)
        :param record_defaults: default value for each column in the csv file if
            one column value is empty in the csv file
        :param shuffle: csv file is read and saved in a queue with method
            tf.train.string-input_producer(). Elements in the queue could be
            shuffled before they are saved. This is recommended when using
            method self.input_shuffle_batch_small_data_sets() (default: False)
        :type shuffle: boolean 
        :return: queue of image path and label from a .csv data list file in tf
            tensor form
        """
        
        # create queue of the image file names and labels by importing the csv
        # file with file names and labels:
        filename_queue = tf.train.string_input_producer(string_tensor=[self.path_csv_file],
                                                        num_epochs=self.num_epochs,
                                                        shuffle=shuffle, 
                                                        seed=None,
                                                        capacity=32,
                                                        shared_name=None,
                                                        name='string_input_producer_'+self.path_csv_file.split('/')[-1])
        # A Reader that outputs the lines of a file delimited by newlines:
        reader = tf.TextLineReader()
        # Now read line by line (key is a string which looks like 
        # 'path_to_file_that_is_read:line_nr', value is the actual read line):
        key, value = reader.read(filename_queue)

        # decode the csv file (each read line from above is splitted by the
        # delimiter ',', a default value is neede):
        # here line is eg: b'train/0/10257_0.png,0'
        return tf.decode_csv(value, record_defaults=record_defaults)
        

    @abc.abstractmethod
    def preprocess_image(self, image, *args, **kwargs):
        pass
    
    def read_image(self, image_path, *args, **kwargs):
        """
        Takes an image path in tf tensor form (from a queue, preferably created
        with method self.read_csv_file()) decodes the image (transforms it to a
        tf tensor object) and does some image preprocessing if necessary.
        ATTENTION: after reading the image from a file, the tensor shape is not
            known and not set in this function.
        :param image_path: path to an image file in tf tensor form
        :type image_path: tf tensor
        :return: preprocessed image in tf tensor form
        """
        # redefine filename as a complete path:
        # The complete path is the path to the .csv file (not including the
        # .csv file, only the directory it is located in) and the
        # continuation given in each line of the .csv file.
        image_path = '/'.join(self.path_csv_file.split('/')[:-1]) + '/' + \
                     image_path

        # Read and outputs the entire contents of the input filename:
        image_file = tf.read_file(image_path, name=None)

        # Decode the image as a JPEG/png file, this will turn it into a Tensor
        # which we can then use in training.
        # Decode a jpg/PNG-encoded image to a uint8 or uint16 tensor.
        # tf.decode_image() uses tf.decode_jpg() or tf.decode_png()
        # automatically, depinding on the input file.
        # attention: tf.image.decode_jpeg/png/image does not get the right
        # shape, because this funciton only adds a node to the graph before
        # seeing the input images -> do not forget to set the correct shape
        image = tf.image.decode_image(contents=image_file#,
                                    # channels=image_shape[-1]  # default:None
                                     )
        image_preprocessed = self.preprocess_image(image, *args, **kwargs)
        return image_preprocessed

    @abc.abstractmethod
    def preprocess_label(self, *args, **kwargs):
        pass

    def read_label(self, *args, **kwargs):
        """
        Takes a label in tf tensor form (from a queue, preferably created with
        method self.read_csv_file()) and does some image preprocessing if
        necessary.
        :param label: tf tensor that holds the label (depending on the task this
            might be a string, file, etc.)
        :type label: tf tensor
        :return: a preprocessed label in tf tensor form
        """
        # If necessary do here a 'read file'...
        
        label_preprocessed = self.preprocess_label(*args, **kwargs)
        return label_preprocessed

    def input_shuffle_batch(self, tensors):
        """
        Takes tf tensor objects images (preferably created with method
        self.read_image()) and labels (preferably created with method
        self.read_csv_file()) and creates shuffled (mini) batches.
        :param tensors: list with tf tensorflow objects that should be
            summarized in a shuffled batch
        :type tensors: list of tf tensor objects
        :return: tf image batches and tf label batches
        """
        # Creates batches of batch_size images and batch_size labels.
        # min_after_dequeue defines how big a buffer we will randomly sample
        # from -- bigger means better shuffling but slower start up and more
        # memory used.
        # capacity must be larger than min_after_dequeue and the amount larger
        # determines the maximum we will prefetch.  Recommendation:
        # min_after_dequeue + (num_threads + a small safety margin) * batch_size
        # 
        # You must ensure that either (i) the shapes argument is passed, or (ii)
        # all of the tensors in tensors must have fully-defined shapes.
        # ValueError will be raised if neither of these conditions holds.
        #
        # http://stackoverflow.com/questions/39283605/regarding-the-use-of-tf-train-shuffle-batch-to-create-batches:
        # Many training algorithms, such as the stochastic gradient
        # descent–based algorithms that TensorFlow uses to optimize neural
        # networks, rely on sampling records uniformly at random from the
        # entire training set. However, it is not always practical to load
        # the entire training set in memory (in order to sample from it),
        # so tf.train.shuffle_batch() offers a compromise: it fills an 
        # internal buffer with between min_after_dequeue and capacity
        # elements, and samples uniformly at random from that buffer. For
        # many training processes, this improves the accuracy of the 
        # model and provides adequate randomization.
        # The min_after_dequeue and capacity arguments have an indirect
        # effect on training performance. Setting a large min_after_dequeue
        # value will delay the start of training, because TensorFlow has to
        # process at least that many elements before training can start. The
        # capacity is an upper bound on the amount of memory that the input
        # pipeline will consume: setting this too large may cause the
        # training process to run out of memory (and possibly start
        # swapping, which will impair the training throughput).
        # 
        # this function has influence on above defined values (eg. on queues),
        # be careful if you want to execute a queue from above and the 
        # following function is defined simultaneously (then not all 
        # iterations through the data set might be possible in the queue).
        # -> iterations through the batches should work like expected
        min_after_dequeue = 5000
        num_threads = 2
        capacity = min_after_dequeue + (num_threads + 1) * self.batch_size
        batch_list = tf.train.shuffle_batch(tensors=tensors,
                                            batch_size=self.batch_size,
                                            num_threads=num_threads,
                                            capacity=capacity,
                                            min_after_dequeue=min_after_dequeue,
                                            name='shuffle_batch_'+self.path_csv_file.split('/')[-1]
                                           )
        return batch_list

    def input_shuffle_batch_small_data_sets(self, tensors):
        """
        Takes tf tensor objects images (preferably created with method
        self.read_image()) and labels (preferably created with method
        self.read_csv_file()) and creates NOT shuffled (mini) batches.
        ATTENTION: use shuffle=True in self.read_csv_file() method.
        :param tensors: list with tf tensorflow objects that should be
            summarized in a shuffled batch
        :type tensors: list of tf tensor objects
        :return: tf image batches and tf label batches
        """
        # http://stackoverflow.com/questions/39283605/regarding-the-use-of-tf-train-shuffle-batch-to-create-batches:
        batch_list = tf.train.batch(tensors=tensors,
                                    batch_size=self.batch_size,
                                    num_threads=1,
                                    capacity=32,
                                    enqueue_many=False,
                                    shapes=None,
                                    dynamic_pad=False,
                                    allow_smaller_final_batch=False,
                                    shared_name=None,
                                    name='shuffle_batch_'+self.path_csv_file.split('/')[-1]
                                   )
        return batch_list

    @abc.abstractmethod
    def next_batch(self, tensor_lst, ohe: bool=True):
        pass

    def set_image_shape(self, image, shape: typing.List[int]):
        """
        sets the shape of an image
        :param image: tf tensor object wich holds an image
        :param shape: Shape of an image in form of [image_height,
            image_width, image_channels]. 
        :type shape: list
        """
        # catch shape related exceptions:
        if len(shape) not in [3]:
            raise ValueError('Shape mismatch. shape must have shape ' +
                             '[image_height, image_width, image_channels]')
        if shape[-1] not in [1, 3]:
            raise ValueError('Number of channels (third entry in ' + 
                             'shape) must be either 1 (greyscale) or 3' +
                             '(rgb)')
        # define the shape of images (all images must have the same dimensions
        # at this point, before batching)
        # shape = [image_height, image_width, image_channels]
        return image.set_shape(shape=shape)


class SintelDataInputQueue(DataInputQueue):

    def __init__(self, *args, **kwargs):
        DataInputQueue.__init__(self, *args, **kwargs)

    def preprocess_image(self,
                         image):
        """
        do some image preprocessing steps.
        in tensorflow there are several methods available
        :param image: tf tensor that holds an image tensor
        :return: tf tensor that holds the preprocessed image.
        """
        # scale image to [0, 1]:
        # image = image / 256
        return image

    def preprocess_label(self, label):
        """
        do some label preprocessing steps.
        :param label: tf tensor that holds the label (depending on the task this
            might be a string, file, etc.)
        :type label: tf tensor
        :return: tf tensor that holds the preprocessed label.
        """
        return label

    def random_crop_images_and_labels(self, image_and_labels, channels,
                                      spatial_shape, data_augmentation=True):
        """
        Randomly crops `image` together with `labels`.
        (idea:
        https://stackoverflow.com/questions/42147427/tensorflow-how-to-randomly-crop-input-images-and-labels-in-the-same-way )
        :param image_and_labels: list of tensors with each shape 
            [height, width, channels]
        :param channels: list that contains the number of corresponding channels
            of list image_and_labels
        :param spatial_shape: tensor/list with shape
            [cropped_height, cropped_width] indicating the crop size (spatial
            dimensions).
        :param data_augmentation: bool, if True data augmentation is performed
            by randomly flipping images horizontally. Because of implementation
            details image rotation cannot be done here but needs to be done
            after batching (see method self.next_batch(). The disadvantage is
            that image rotation is performed after cropping is done. This leads
            to more areas in the images that have pixel values 0 (default: True)
        :returns: list of tensors (same dimension as input image_and_labels) of
            cropped images labels.
        """
        assert len(image_and_labels) == len(channels)
        # concat by axis=-1 ^= channels (stack images in channel dimension):
        combined_img = tf.concat(image_and_labels, axis=-1)
        try:
            # get a list of all channels, necessary for splitting images later:
            nr_channels_lst = [0]
            # total count of all channels:
            nr_channels = 0
            for image in image_and_labels:
                nr_channels_lst = tf.concat([nr_channels_lst,
                                             [tf.shape(image)[-1]]], axis=0)
                nr_channels += tf.shape(image)[-1]        

            # cummulated channel numbers for separating images again:
            nr_channels_lst = tf.cumsum(nr_channels_lst)
        except TypeError:
            # a TypeError occures if the shape of an image tensor is unknown
            nr_channels = np.array(channels).sum()
            nr_channels_lst = np.array([0] + channels).cumsum()


        # scale image to [0, 1]:
        combined_img = combined_img / 256

        if data_augmentation:
            # randomly mirror image horizontally:
            combined_img = tf.image.random_flip_left_right(image=combined_img)

        # randomly crop all stacked images:
        combined_crop = tf.random_crop(value=combined_img,
                                       size=tf.concat([spatial_shape,
                                                       [nr_channels]],
                                                      axis=0),
                                       seed=None,
                                       name='crop')
        # split stacked images again for output:
        return_lst = []
        for i in range(len(image_and_labels)):
            image_splitted = combined_crop[:, :,
                                           nr_channels_lst[i]:nr_channels_lst[i+1]]

            self.set_image_shape(image=image_splitted, 
                                 #shape=tf.concat([spatial_shape,
                                 #                 channels[i]],
                                 #                axis=0))
                                 shape=spatial_shape+[channels[i]])
            return_lst += [image_splitted]
        return return_lst

    def next_batch(self, image_shape, data_augmentation=False):
        """
        A 'summary' method that uses other methods of the same  class to read
        image files and their labels from a .csv data list file, decodes them
        and summarizes them in (mini) batches.
        :param image_shape: shape of the cropped image in form of [image_height,
            image_width, image_channels]. Every output image (input image,
            albedo label and shading label) will have this specified dimension.
        :param data_augmentation: bool, if True data augmentation is performed
            by randomly flipping images horizontally and rotating images.
            Because of implementation details image rotation needs to be done
            after batching (in this method). The disadvantage is
            that image rotation is performed after cropping is done. This leads
            to more areas in the images that have pixel values 0 (default: True)
        :return: tf image batches and tf label batches
        """
        image_path, albedo_label_path, \
                shading_label_path = self.read_csv_file(shuffle=True,
                                                        record_defaults=[[''],
                                                                         [''],
                                                                         ['']])
        images = self.read_image(image_path=image_path)
        labels_albedo = self.read_image(image_path=albedo_label_path)
        labels_shading = self.read_image(image_path=shading_label_path)

        images, labels_albedo, labels_shading = \
                self.random_crop_images_and_labels(image_and_labels=[images, 
                                                                     labels_albedo,
                                                                     labels_shading],
                                                   channels=[image_shape[-1]]*3,
                                                   spatial_shape=image_shape[:2],
                                                   data_augmentation=data_augmentation)
        
#        return self.input_shuffle_batch(tensors=[image_path, albedo_label_path,
#                                                 shading_label_path, images,
#                                                 labels_albedo, labels_shading])

        return_lst = self.input_shuffle_batch_small_data_sets(tensors=[image_path,
                                                                       albedo_label_path,
                                                                       shading_label_path,
                                                                       images,
                                                                       labels_albedo,
                                                                       labels_shading])
        image_path_batch, albedo_label_path_batch, shading_label_path_batch, \
            images_batch, labels_albedo_batch, labels_shading_batch = return_lst

        # here we rotate images. we must do it here after batching because of
        # how the function tf.contrib.image.rotate is implemented (especially in
        # comparison to function tf.image.random_flip_left_right:
        if data_augmentation:
            # we chose a rotation angle between [-15, +15] degree = [-.26, .26]
            # rad for data augmentation. However, since we prefer more small
            # rotation angles, we a normal distribution with appropriate scale
            # (std):
            # angle = np.random.uniform(low=-.26, high=0.26)
            angle = np.random.normal(loc=0, size=self.batch_size, scale=0.26/2)
            # rotate image by amount angle:
            images_batch = tf.contrib.image.rotate(images=images_batch,
                                                   angles=angle)
            labels_albedo_batch = tf.contrib.image.rotate(images=labels_albedo_batch,
                                                          angles=angle)
            labels_shading_batch = tf.contrib.image.rotate(images=labels_shading_batch,
                                                           angles=angle)

        return (image_path_batch,
                albedo_label_path_batch,
                shading_label_path_batch,
                images_batch,
                labels_albedo_batch,
                labels_shading_batch)


class DataQueue:
    """
    Base class of ImageQueue, which constructs a shuffled queue of image paths
    (strings) from a given directory without replacement. Once the queue is
    "exhausted", the isExhausted flag returns False, indicating the end of an
    epoch.  The queue then automatically refills itself. "Exhausted" refers to
    the point where dequeue() is called and there are not enough images left in
    the queue after that to satisfy the batch size,
    The remaining images are therefore ignored.
    :param name: Name of the queue.
    :param dir: Directory containing the images.
    :param batch_size: The batch size to be used during training.

    Variables:
    :param base: The list of image paths. Stays constant after constructed.
    :param queue: The queue (list) of image paths.

    Note: This user-friendly custom queue has several advantages over the
    regular Tensorflow method (for example in the CIFAR-10 example code), where
    two queues are required (a string_input_producer which returns image paths
    and a batch shuffler that stores a definable number of images as 4D Tensors
    and returns a batch in each training step). In contrast, this queue:
    1. simply stores and outputs a list of image paths. These strings are then
       read by read_image_batch_png() in each training step, so that space for
       only one image batch Tensor needs to be preallocated, saving precious
       memory.
    2. returns a flag to indicate the end of an epoch instead of an Exception.
       More importantly, it is able to refill itself, whereas Tensorflow Queues
       do not once the desired number of epochs are achieved.
    The slight disadvantage is that the queue has to be explicitly dequeued and
    the strings have to be fed into the model in each training step.
    """

    def __init__(self, df, batch_size, num_epochs):
        self.df = df
        # build the shuffled starting queue:
        self.df_queue = self.df.sample(frac=1)
        # batch size determines how many rows of the DataFrame are droped at
        # once in method dequeue:
        self.batch_size = batch_size
        self.completed_epochs = 0
        self.num_epochs = num_epochs
        self.iter_left = int(self.df.shape[0] / self.batch_size) * self.num_epochs
        self.num_iter = 0

    def dequeue(self):
        """
        When called once, returns:
            batch: A list of image paths according to batch_size.
            isExhausted: Flag indicating if an epoch is achieved.
        """
        # check if we already reached one full epoch (^= all data is dequeued
        # once):
        if ((self.df_queue.shape[0] < self.batch_size) and
                (self.completed_epochs <= self.num_epochs)):
            # then start a new epoch
            self.df_queue = self.df.sample(frac=1)
            self.completed_epochs += 1

        if (self.completed_epochs + 1) > self.num_epochs:
            raise IndexError('{} complete '.format(self.num_epochs) +
                             'epochs have already been dequeued. No more ' +
                             'data available')

        df_batch = self.df_queue.tail(self.batch_size)
        self.df_queue.drop(df_batch.index, inplace=True)
        # lower number of available iterations:
        self.iter_left -= 1
        self.num_iter += 1
        return df_batch


def image_random_flip(image):
    """
    Takes an image-like numpy array and flips the image horizontally with
    probability 0.5.
    :param image: an image-like numpy array with shape [height, width, channels]
        image-like means that we can use in-depth stacked images (eg. channels=9
        for 3 stacked images) to perform the same processing steps
        simultaneously on all stacked images.
    :type image: np.array of shape [height, width, channels]
    :return: randomly flipped np.array image
    """
    # flip image horizontally randomly:
    # bool random number which decides if images should be flipped
    # horizontally:
    is_flip = np.random.choice([False, True], p=[0.5, 0.5])
    # flip image:
    return image[:, ::-1]


def image_random_rotate(image):
    """
    Takes an image-like numpy array and rotates it spatially by a random angle 
    between [-15, 15] deg.
    :param image: an image-like numpy array with shape [height, width, channels]
        image-like means that we can use in-depth stacked images (eg. channels=9
        for 3 stacked images) to perform the same processing steps
        simultaneously on all stacked images.
    :type image: np.array of shape [height, width, channels]
    :return: randomly rotated np.array image
    ATTENTION: image rotation (function sp.ndimage.interpolation.rotate() and
        underlying function sp.ndimage._nd_image.geometric_transform() might
        have a 'long' run-time)
    """
    # rotate image by angle degrees:
    # angle in degrees:
    angle = np.random.normal(loc=0, scale=15 / 2)
    return sp.ndimage.interpolation.rotate(input=image, angle=angle,
                                            axes=(1, 0), reshape=False,
                                            output=None, order=3,
                                            mode='constant', cval=0.0,
                                            prefilter=True)


def image_random_crop(image, output_shape):
    """
    Takes an image-like numpy array and crops it to the size of output_shape at
    a random spatial position.
    :param image: an image-like numpy array with shape [height, width, channels]
        image-like means that we can use in-depth stacked images (eg. channels=9
        for 3 stacked images) to perform the same processing steps
        simultaneously on all stacked images.
    :type image: np.array of shape [height, width, channels]
    :param output_shape: Spatial output shape of the image/stacked images.
    :type output_shape: If output_shape=None (default) the output image/stacked
        images have the same shape as the input image, otherwise output_shape
        must be of type list with at least len(output_shape)==2 elements. The
        output image will have shape
        [output_shape[0], output_shape[1], channels] (where channels = input
        channels)
    :return: randomly cropped np.array image
    """
    # randomly crop image to output shape:
    if image.shape[0]!=output_shape[0]:
        y_start = np.random.randint(image.shape[0] - output_shape[0])
    else:
        y_start = 0
    if image.shape[1]!=output_shape[1]:
        x_start = np.random.randint(image.shape[1] - output_shape[1])
    else:
        x_start = 0
    return image[y_start:y_start + output_shape[0],
                 x_start:x_start + output_shape[1], :]


def image_normalize(image):
    """
    Takes an image-like numpy array and normalizes it (by dividing each pixel
    throug 256). So, images are normalized to range [0, 1].
    :param image: an image-like numpy array with shape [height, width, channels]
        image-like means that we can use in-depth stacked images (eg. channels=9
        for 3 stacked images) to perform the same processing steps
        simultaneously on all stacked images.
    :type image: np.array of shape [height, width, channels]
    :return: normalized np.array image
    """
    return image / 256


def image_scale(image, random):
    """
    Scales an image by a random (relative) factor between 0.9 and 1.1.
    :param image: an image [height, width, channels]
    :type image: np.array of shape [height, width, channels]
    :return: processed np.array image
    """
    return sp.misc.imresize(arr=image, size=rand_scale,
                            interp='bilinear', mode=None)

    
def next_batch(deq, output_shape=None, is_scale=True, is_flip=True,
               is_rotated=True, norm=True):
    """
    Generates a new processed batch of images and labels each time it is
    called (if a DataQueue.dequeue() object is passed).
    :param deq: typically a DataQueue.dequeue() (format pd.DataFrme!!!) object
        which outputs a batch of data in form of a pd.DataFrame() which contains
        paths of all images and labels.
    :type deq: DataQueue.dequeue() object
    :param output_shape: Spatial output shape of the image/stacked images.
    :type output_shape: If output_shape=None (default) the output image/stacked
        images have the same shape as the input image (if is_scale==False),
        otherwise output_shape must be of type list with at least
        len(output_shape)==2 elements. The output image will have shape
        [output_shape[0], output_shape[1], channels] (where channels = input
        channels)
    :param is_scale: scale image by a random (relative) factor between 0.9 and
        1.1. It is most advisable to only scale images if the out_shape is
        significantly smaller than the input images, because if the image
        scaling results in images smaller than the desired output, cropping is
        not possible.
        To prevent too small scaled images, images are only scaled if the
        scaling results in output images >= output_shape
    :type is_scale: bool (default: True)
    :param is_flip: flag for randomly flipping image horizontally
    :type is_flip: bool (default: True)
    :param is_rotated: flag for randomly rotating image by random angle in range
        [-15, 15] deg
    :type is_rotated: bool (default: True)
    :param norm: norm images to [0, 1] range.
    :type norm: boolean (default: True)
    :return: batch of images, albedo labels and shading labels
    ATTENTION: image rotation (function sp.ndimage.interpolation.rotate() and
        underlying function sp.ndimage._nd_image.geometric_transform() take
        by far most of the time)
    """
    batches = []
    for i in deq:
        batches.append([])

    for row in deq.values:
        # read images (image + labels):
        imgs = [sp.misc.imread(name=path, flatten=False,
                               mode='RGB') for path in row]

        # if is_scale randomly scale images by factor between 0.9 and 1.1:
        if is_scale and not output_shape:
            raise ValueError('If is_scale=True but output_shape=None this ' +
                             'leads to an inconsistency error. To create ' +
                             'batches each image in the batch must have the ' +
                             'same shape.')
        elif is_scale and output_shape:
            scale_min = 0.9
            scale_max = 1.1
            # scale images in batch with a random scale factor between 0.9
            # and 1.1:
            rand_scale = np.random.uniform(low=scale_min, high=scale_max, 
                                           size=None)
            # if the scaling gets smaller than the wanted cropping output
            # (defined by output_shape) than skip scaling (-> rand_scale=1.0):
            if np.any(np.greater(np.array(output_shape[:-1]), rand_scale * np.array(np.array(imgs).shape[1:3]))):
                rand_scale = 1.0

            for i, img in enumerate(imgs):
                imgs[i] = sp.misc.imresize(arr=img, size=rand_scale,
                                           interp='bilinear', mode=None)

        # stack images and labels along depth (channels) to perform the same
        # operations on them:
        imgs_stacked = np.dstack(imgs)

        # preprocess images (randomly flip horizontally, rotate and/or crop):
        if is_flip:
            imgs_stacked = image_random_flip(image=imgs_stacked)
        if is_rotated:
            imgs_stacked = image_random_rotate(image=imgs_stacked)
        if output_shape:
            imgs_stacked = image_random_crop(image=imgs_stacked,
                                             output_shape=output_shape)
        if norm:
            imgs_stacked = image_normalize(image=imgs_stacked)
        # split images to get [image, albedo, shading]:
        split_points = np.array([img.shape[2] for img in imgs]).cumsum()[:-1]
        imgs_splitted = np.split(ary=imgs_stacked,
                                 indices_or_sections=split_points,
                                 axis=2)
        for i, col in enumerate(deq):
            batches[i].append(imgs_splitted[i])
    return (np.stack(batch) for batch in batches)

