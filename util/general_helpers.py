#!/usr/bin/env python                            
                                                 
"""                                              
General smaller, usefull helper functions.
"""                                              


__author__ = "Udo Dehm"                          
__copyright__ = "Copyright 2017"                 
__credits__ = ["Udo Dehm"]                       
__license__ = ""                                 
__version__ = "0.1"                              
__maintainer__ = "Udo Dehm"                      
__email__ = "udo.dehm@mailbox.org"               
__status__ = "Development"                       
                                                 
                                                 
__all__ = ['get_time_format', 'time_tuple_to_str']   


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

