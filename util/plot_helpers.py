#!/usr/bin/env python                            
                                                 
"""                                              
Usefull functions for plotting
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
                                                 
                                                 
__all__ = ['plot_images']   


def plot_images(images, titles='', *args, **kwargs):
    """
    Show images
    :param images: list of images
    :param titles: list of titles or string with one title, this title is then
        used for all images (default: '')
    :return: nothing (only side effects)
    """
    if isinstance(titles, str):
        titles = [titles] * len(images)

    assert len(images)==len(titles)

    for image, title in zip(images, titles):
        plt.figure(*args, **kwargs)
        plt.title(title)
        plt.imshow(image)
