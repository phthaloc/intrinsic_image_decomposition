# README #

Deep Intrinsic Image Decomposition:

### What is this repository for? ###

* Quick summary
* 0.1


### How do I get set up? ###

* clone the repository 
* create the needed data sets with data_structure.py.
This script creates by default a data/ sub-directory where the data and csv files (training, validation, testing set definitions) are stored.
* own models are defined in cnn_model.py
* utility (helper) functions are defined in module utility
* jupyter file cnns.ipynb creates complete models and trains them
    * it uses script input_queues.py where data input queues are created.
    * (trained) tensorflow models and parameter data is saved in sub-directory logs/ (this sub-directory is created during training)
* jupyter file data_analysis.ipynb analyses the data (image sizes etc.)
*