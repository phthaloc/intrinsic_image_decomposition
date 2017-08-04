#!/usr/bin/env python                            
                                                 
"""                                              
General smaller, usefull helper functions.
"""                                              

import logging

__author__ = "Udo Dehm"                          
__copyright__ = "Copyright 2017"                 
__credits__ = ["Udo Dehm"]                       
__license__ = ""                                 
__version__ = "0.1"                              
__maintainer__ = "Udo Dehm"                      
__email__ = "udo.dehm@mailbox.org"               
__status__ = "Development"                       
                                                 
                                                 
__all__ = ['get_time_format', 'time_tuple_to_str', 'create_logger']   


def get_time_format(time_in_sec):
    """
    Outputs a time range (in sec) in hours, minutes and seconds
    :param time_in_sec: a time range in seconds
    :type time_in_sec: float
    :returns: tuple (int(hours), int(minutes), int(seconds))
    """
    hours = int(time_in_sec // 3600)
    mins = int(time_in_sec % 3600 // 60)
    secs = int(time_in_sec % 3600 % 60)
    return hours, mins, secs

def time_tuple_to_str(time_tuple):
    s = '{h:02}:{m:02}:{s:02}'.format(h=time_tuple[0], m=time_tuple[1],
                                      s=time_tuple[2])
    if time_tuple[0]!=0:
        s += ' h'
    else:
        if time_tuple[1]!=0:
            s = s[3:] + ' min'
        else:
            s = s[6:] + ' sec'
            s = s.lstrip('0')
    return s

def create_logger(filename):
    """
    Creates a logger for logging outputs.
    This logger writes messages both to stdout and a file.
    :param filename: filename of the log file.
    :type filename: str
    :returns: logger object
    """
    # create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('[%(asctime)s %(levelname)s]: %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    fh = logging.FileHandler(filename, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
