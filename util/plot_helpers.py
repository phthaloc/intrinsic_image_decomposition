#!/usr/bin/env python                            
                                                 
"""                                              
Usefull functions for plotting
"""                                              

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output, Image, display, HTML

__author__ = "ud"
__copyright__ = "Copyright 2017"
__credits__ = ["phthalo@mailbox.org"]
__license__ = ""
__version__ = "0.1"
__maintainer__ = "ud"
__email__ = "phthalo@mailbox.org"
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


# the following two functions are for plotting a tensorflow graph in a jupyter
# notebook file, see
# https://stackoverflow.com/questions/38189119/simple-way-to-visualize-a-tensorflow-graph-in-jupyter
def _strip_consts(graph_def, max_const_size=32):
    """
    Strip large constant values from graph_def.
    """
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                try:
                    tensor.tensor_content = "<stripped %d bytes>"%size
                except TypeError:
                    tensor.tensor_content = bytes("<stripped %d bytes>"%size,
                                                  'utf-8')
    return strip_def


def show_graph(graph_def, max_const_size=32): 
    """
    Visualize TensorFlow graph.
    """
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    
    strip_def = _strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
            function load() {{
                document.getElementById("{id}").pbtxt = {data};
            }}
        </script> 
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
            <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))
    
    iframe = """ 
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))

