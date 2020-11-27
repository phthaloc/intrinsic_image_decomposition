# Deep Intrinsic Image Decomposition

## Abstract
This project addresses the problem of decomposing a single RGB image into its
intrinsic components -- the albedo or reflectance layer and the shading layer.
We use state of the art deep learning architectures based on fully
convolutional neural networks to predict the albedo and shading images in a
regression learning task. The models are designed in such a way that the
predictions can be made directly, without post-processing or similar additional
steps.

Since the intrinsic image decomposition problem is an ill-posed problem we have
to impose some constraints to achieve a successful learning process.
We do this by using labeled datasets in a supervised learning task. Therefore we
heavily depend on the availability of these datasets.
There are two suitable datasets publicly available: the synthetic [MPI Sintel dataset](http://sintel.is.tue.mpg.de/) and the
[Intrinsic Images in the Wild dataset](http://opensurfaces.cs.cornell.edu/publications/intrinsic/) which contains real-world scenes.

Once our models have been trained they can be instantly employed and used for
real-time predictions on images or videos.

We compare different network architecture styles and loss functions.

On the Sintel dataset we are able to deliver state-of-the-art results comparable
or better than privious work. This approach shows that CNN-based models can be
successfully trained on this synthetic dataset.

On the Intrinsic Images in the Wild dataset we get less significant results.
This seems to be because of the purely data-driven approach (sparse labeling) and the special
construction of the loss function required for this dataset.

## Introduction
Intrinsic image decomposition is the decomposition of a color image `i` into its components albedo layer `a` (reflectance layer) and shading layer `s`. They are connected by a pixe-wise product
```
i = a * s
```

What are the benefits of decomposing images into its intrinsic layers?
- easier interpretation of scenes
- simple extraction of the geometry of an object
- segmentation of objects is straight forward
- material recognition in images
- resourfacing images


## Models

<img src="imgs/decoder_narihira2015_reduced.png" width=400em>

<img src="imgs/mpi_sintel_dataset-example.png" width=700em>
<img src="imgs/network_architecture_coarse_sintel.png" width=500em>


<img src="imgs/bell2014_dataset-example.png" width=700em>

<img src="imgs/network_architecture_coarse_iiw.png" width=700em>


## Results
<img src="imgs/predictions_best_model.png" width=700em>

<img src="imgs/predictions_comparison_sintelmodel_on_iiw.png" width=500em>


## How to use the scripts

* clone the repository 
* create the needed data sets with script `util/data_structure.py`.
This script creates by default a `data/` sub-directory where the data and csv files (training, validation, testing set definitions) are stored.
* custom models are defined in `scripts_cnn_models_slim/cnn_model.py`
* further utility (helper) functions are defined in sub-directory `util/`
* the jupyter notebook `cnns.ipynb` (`cnns.py`) creates complete models and trains them
    * it uses script `input_queues.py` where data input queues are created.
    * (trained) tensorflow models and parameter data are saved in sub-directory `logs/` (this sub-directory is created during training)
* jupyter file `data_analysis.ipynb` analyses the data (image sizes etc.)
* `cnns_predict.ipynb` Inference script shows how to programm and setup a convolutional neural network using Tensorflow.
* (script `create_models.ipynb` creates graphs and saves them to the appropriate location)
* (`iiw_visualization_whdr_metrics.ipynb` analyzes the WHDR metric (for more information see `whitepaper.pdf`))
* (script `loss_function_analysis.ipynb` analyzes loss functions of different models)
* (The script `cnns_tf_input_queue.ipynb` imports inference graphs and extends them to training graphs)


