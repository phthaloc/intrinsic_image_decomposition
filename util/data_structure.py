#!/usr/bin/env python

"""
# Create data file structures and sets:

    This script creates csv files with file names and labels for all used data
    sets.
    The tensorflow pipeline reads the csv files later and imports images in
    batches.

    Used data sets:
        - intrinsic imigas in the wild (iiw)
        - MPI Sintel data set
        - MIT data set
"""

import sys

sys.path.append('./util')
import os
import glob
import pandas as pd
import scipy as sp
import scipy.misc
import download

__author__ = "Udo Dehm"
__copyright__ = "Copyright 2017"
__credits__ = ["Udo Dehm"]
__license__ = ""
__version__ = "0.1"
__maintainer__ = "Udo Dehm"
__email__ = "udo.dehm@mailbox.org"
__status__ = "Development"

__all__ = ['', 'main']


def create_datasets_mit(df, p_train, p_valid, p_test, sample=True):
    """
    Splits a data set df into training, validation and testing data set with
    relative cardinality p_train, p_valid and p_test, respectively.
    :param df: complete data set which should be split into training, validation
        and testing sets
    :type df: pd.DataFrame()
    :param p_train: relative cardinality of training data set
    :type p_train: float (\elem [0,1])
    :param p_valid: relative cardinality of validation data set
    :type p_valid: float (\elem [0,1])
    :param p_test: relative cardinality of testing data set
    :type p_test: float (\elem [0,1])
    :return: training, validation and testing data sets
    :type: [pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    # make sure we have consistancy:
    assert p_train + p_valid + p_test  == 1, 'p_train, p_valid, p_test must add up to 1'
    # this data set will be the training data set in the end:
    df_train = df.copy()
    # sampling data to get testing set:
    df_test = df_train.sample(n=int(p_test * df.shape[0]), 
                                    frac=None,
                                    replace=False,
                                    weights=None,
                                    random_state=42,
                                    axis=0)
    # drop these sampled test data (we do not want them in the other data sets):
    df_train.drop(df_test.index, inplace=True)
    # sampling data to get validation set:
    df_valid = df_train.sample(n=int(p_valid * df.shape[0]), frac=None,
                               replace=False, weights=None, random_state=42,
                               axis=0)
    # drop these sampled valid data (we do not want them in the training set):
    df_train.drop(df_valid.index, inplace=True)

    if sample:
        # now create sample files:
        df_train_sample = df_train.sample(n=50, frac=None, replace=False,
                                          weights=None, random_state=42, axis=0)
        df_valid_sample = df_valid.sample(n=20, frac=None, replace=False,
                                          weights=None, random_state=42, axis=0)
        df_test_sample = df_test.sample(n=20, frac=None, replace=False,
                                        weights=None, random_state=42, axis=0)
    else:
        df_train_sample = pd.DataFrame()
        df_valid_sample = pd.DataFrame()
        df_test_sample = pd.DataFrame()
    # print info:
    print('data set cardinalities:\n' +
          '    # complete data set: {}\n'.format(len(df)) +
          '    # training data set: {}\n'.format(len(df_train)) +
          '    # validation data set: {}\n'.format(len(df_valid)) +
          '    # testing data set: {}\n'.format(len(df_test)) +
          '    # sample training data set: {}\n'.format(len(df_train_sample)) +
          '    # sample validation data set: {}\n'.format(len(df_valid_sample))+
          '    # sample testing data set: {}'.format(len(df_test_sample))
          )
    return (df_train, df_valid, df_test, df_train_sample, df_valid_sample,
            df_test_sample)


def create_datasets_iiw(df, data_dir, p_train, p_valid, p_test, sample=True):
    """
    Splits a data set df into training, validation and testing data set with
    relative cardinality p_train, p_valid and p_test, respectively.
    :param df: complete data set which should be split into training, validation
        and testing sets
    :type df: pd.DataFrame()
    :param p_train: relative cardinality of training data set
    :type p_train: float (\elem [0,1])
    :param p_valid: relative cardinality of validation data set
    :type p_valid: float (\elem [0,1])
    :param p_test: relative cardinality of testing data set
    :type p_test: float (\elem [0,1])
    :return: training, validation and testing data sets
    :type: [pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    # make sure we have consistancy:
    assert p_train + p_valid + p_test == 1, 'p_train, p_valid, p_test must add up to 1'

    # read every file saved in column file_path and get shape:
    df['image_shape'] = (data_dir + df['image_path']).apply(lambda x: sp.misc.imread(x).shape)
    # expand height, width and nr of channels to separate rows
    df['image_shape'] = df['image_shape'].apply(lambda x: x + (1,) if len(x) == 2 else x)
    df[['image_height', 'image_width', 'image_nr_channels']] = df['image_shape'].apply(pd.Series)

    print('Deleted {} '.format(df[(df['image_height'] < 340) | (df['image_width'] < 340)].shape[0]) +
          'images because of too small image height or width')

    # delete images that are too small:
    df = df[(df['image_height'] >= 340) & (df['image_width'] >= 340)]

    # this data set will be the training data set in the end:
    df_train = df.copy()

    # sampling data to get testing set:
    df_test = df_train.sample(n=int(p_test * df.shape[0]),
                              frac=None,
                              replace=False,
                              weights=None,
                              random_state=42,
                              axis=0)

    # drop these sampled test data (we do not want them in the other data sets):
    df_train.drop(df_test.index, inplace=True)
    # sampling data to get validation set:
    df_valid = df_train.sample(n=int(p_valid * df.shape[0]), frac=None,
                               replace=False, weights=None, random_state=42,
                               axis=0)
    # drop these sampled valid data (we do not want them in the training set):
    df_train.drop(df_valid.index, inplace=True)

    if sample:
        # now create sample files:
        df_train_sample = df_train.sample(n=50, frac=None, replace=False,
                                          weights=None, random_state=42, axis=0)
        df_valid_sample = df_valid.sample(n=20, frac=None, replace=False,
                                          weights=None, random_state=42, axis=0)
        df_test_sample = df_test.sample(n=20, frac=None, replace=False,
                                        weights=None, random_state=42, axis=0)
    else:
        df_train_sample = pd.DataFrame()
        df_valid_sample = pd.DataFrame()
        df_test_sample = pd.DataFrame()
    # print info:
    print('data set cardinalities:\n' +
          '    # complete data set: {}\n'.format(len(df)) +
          '    # training data set: {}\n'.format(len(df_train)) +
          '    # validation data set: {}\n'.format(len(df_valid)) +
          '    # testing data set: {}\n'.format(len(df_test)) +
          '    # sample training data set: {}\n'.format(len(df_train_sample)) +
          '    # sample validation data set: {}\n'.format(len(df_valid_sample)) +
          '    # sample testing data set: {}'.format(len(df_test_sample))
          )
    return (df_train, df_valid, df_test, df_train_sample, df_valid_sample,
            df_test_sample)


def create_datasets_sintel(df):
    """
    Splits a data set df into training, validation and testing data set.
    :param df: complete data set which should be split into training, validation
        and testing sets
    :type df: pd.DataFrame()

    """
    # for sintel data set split training/validation/testing
    # sets by scene:
    sintel_scenes = dict(
        train=['alley_1', 'alley_2', 'ambush_2', 'ambush_4',
               'ambush_5', 'ambush_6', 'ambush_7', 'bamboo_1',
               'bandage_1', 'bandage_2', 'cave_2', 'cave_4',
               'market_2', 'market_6', 'shaman_2', 'shaman_3',
               'sleeping_1', 'sleeping_2', 'temple_2', 'temple_3'],
        valid=['bamboo_2', 'market_5', 'mountain_1'])
    # get training validation and testing data set of the mpi-sintel
    # data:
    # create training set by filtering for 'training' scenes:
    df_train = df[df['scene_c'].isin(sintel_scenes['train'])]
    # delete irrelevant columns and shuffle data set randomly:
    df_train = df_train.drop('scene_c', axis=1).sample(frac=1)
    # create validation set by filtering for 'validation' scenes:
    df_valid_filtered = df[df['scene_c'].isin(sintel_scenes['valid'])]
    # delete irrelevant columns and shuffle data set randomly:
    df_valid = df_valid_filtered.drop('scene_c', axis=1).sample(frac=1)
    # create test set from validation set by randomly sampling 5 elements
    # from each of the validation scene:
    df_test = df_valid_filtered.groupby('scene_c').apply(lambda x: x.sample(n=16))
    # delete irrelevant columns
    df_test.drop('scene_c', axis=1, inplace=True)
    # delete multi index:
    df_test.index = df_test.index.droplevel(level=0)
    # delete rows from validation set that are now in testing set:
    df_valid = df_valid[~df_valid.index.isin(df_test.index)]

    # get sample training data set:
    df_train_sample = df_train.sample(n=50, frac=None, replace=False,
                                      weights=None, random_state=42, axis=0)
    # get sample validation data set:
    df_valid_sample = df_valid.sample(n=20, frac=None, replace=False,
                                      weights=None, random_state=42, axis=0)
    # get sample testing data set (same as testing set because of low # of data):
    df_test_sample = df_test
    return df_train, df_valid, df_test, df_train_sample, df_valid_sample, df_test_sample


def main(data_set, data_dir='data/', create_csv_lists=True):
    # urls for all data sets:
    url_iiw = 'http://labelmaterial.s3.amazonaws.com/release/iiw-dataset-release-0.zip'

    url_sintel_complete = 'http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip'

    url_sintel_albedo = 'http://files.is.tue.mpg.de/jwulff/sintel/extras/MPI-Sintel-training_albedo_noshadingtextures.zip'
    url_sintel_images = 'http://files.is.tue.mpg.de/jwulff/sintel/extras/MPI-Sintel-training_clean_noshadingtextures.zip'
    url_sintel_shading = 'http://files.is.tue.mpg.de/jwulff/sintel/extras/MPI-Sintel-training_shading.zip'

    url_mit = 'http://people.csail.mit.edu/rgrosse/intrinsic/intrinsic-data.tar.gz'

    # directory of data
    data_dir_sintel_complete = data_dir + 'mpi-sintel-complete/'
    data_dir_sintel_shading = data_dir + 'mpi-sintel-shading/'
    data_dir_mit = data_dir + 'MIT-intrinsic/data/'
    data_dir_iiw = data_dir + 'iiw-dataset/data/'

    if data_set == 'iiw':
        download.maybe_download_and_extract(url=url_iiw, download_dir=data_dir)

        if create_csv_lists:
            # import file names of data directory:
            df_iiw = pd.DataFrame([[int(os.path.splitext(os.path.basename(x))[0]),
                                    os.path.relpath(x, data_dir),
                                    os.path.splitext(os.path.relpath(x, data_dir))[0] + '.json'] for x in
                                   glob.glob(data_dir_iiw + '/*.png')],
                                  columns=['file_id', 'image_path', 'label_path'])
            # sort by file ids (we can sort these files because they are
            # shuffled during training in tf anyways):
            df_iiw.sort_values(by='file_id', inplace=True)
            # reset indices of pd.DataFrame:
            df_iiw.reset_index(drop=True, inplace=True)

            # get training validation and testing data set of the iiw data:
            df_iiw_train, df_iiw_valid, df_iiw_test, df_iiw_train_sample, \
            df_iiw_valid_sample, \
            df_iiw_test_sample = create_datasets_iiw(df=df_iiw, p_train=0.8,
                                                     data_dir=data_dir,
                                                     p_valid=0.1,
                                                     p_test=0.1,
                                                     sample=True)

            # save complete data set, training data set, validation data set and
            # testing data set in separate data files:
            df_iiw.to_csv(path_or_buf=data_dir + 'data_iiw_complete.csv',
                          sep=',', columns=['image_path', 'label_path'],
                          index=False, header=False)
            df_iiw_train.to_csv(path_or_buf=data_dir + 'data_iiw_train.csv',
                                sep=',', columns=['image_path', 'label_path'],
                                index=False, header=False)
            df_iiw_valid.to_csv(path_or_buf=data_dir + 'data_iiw_valid.csv',
                                sep=',', columns=['image_path', 'label_path'],
                                index=False, header=False)
            df_iiw_test.to_csv(path_or_buf=data_dir + 'data_iiw_test.csv',
                               sep=',', columns=['image_path', 'label_path'],
                               index=False, header=False)

            df_iiw_train_sample.to_csv(path_or_buf=data_dir + 'sample_data_iiw_train.csv',
                                       sep=',',
                                       columns=['image_path', 'label_path'],
                                       index=False, header=False)
            df_iiw_valid_sample.to_csv(path_or_buf=data_dir + 'sample_data_iiw_valid.csv',
                                       sep=',',
                                       columns=['image_path', 'label_path'],
                                       index=False, header=False)
            df_iiw_test_sample.to_csv(path_or_buf=data_dir + 'sample_data_iiw_test.csv',
                                      sep=',',
                                      columns=['image_path', 'label_path'],
                                      index=False, header=False)

    elif data_set == 'mpi_sintel_complete':
        data_dir_sintel_complete = data_dir + 'mpi-sintel-complete/'
        download.maybe_download_and_extract(url=url_sintel_complete,
                                            download_dir=data_dir_sintel_complete)
        if create_csv_lists:
            # use 'clean pass' images (see narihira2015: p.3: "'final images'
            # [...] are the result of additional computer graphics tricks which
            # dristract from our application."):
            df_sintel = pd.DataFrame([[os.path.relpath(x, data_dir),
                                       os.path.relpath(x, data_dir).replace('clean', 'albedo'),
                                       os.path.relpath(x, data_dir).replace('clean', 'invalid')
                                       ] for x in glob.glob(data_dir_sintel_complete + 'training/clean/**/*.png')],
                                     columns=['image_path', 'label_path', 'invalid_path'])
            # add scene to dataframe (for splitting into train/valid/test sets):
            df_sintel['scene_c'] = df_sintel['image_path'].apply(lambda row: row.split('/')[-2:-1]).apply(pd.Series)

            # get training validation and testing data set of the mpi-sintel
            #  data:
            df_sintel_train, df_sintel_valid, df_sintel_test, df_sintel_train_sample, \
                df_sintel_valid_sample, df_sintel_test_sample = create_datasets_sintel(df_sintel)

            # save complete data set, training data set, validation data set and
            # testing data set in separate data files:

            df_sintel.to_csv(path_or_buf=data_dir + 'data_sintel_complete_complete.csv',
                             sep=',',
                             columns=['image_path', 'label_path', 'invalid_path'],
                             index=False,
                             header=False)
            df_sintel_train.to_csv(path_or_buf=data_dir + 'data_sintel_complete_train.csv',
                                   sep=',',
                                   columns=['image_path', 'label_path', 'invalid_path'],
                                   index=False,
                                   header=False)
            df_sintel_valid.to_csv(path_or_buf=data_dir + 'data_sintel_complete_valid.csv',
                                   sep=',',
                                   columns=['image_path', 'label_path', 'invalid_path'],
                                   index=False,
                                   header=False)
            df_sintel_test.to_csv(path_or_buf=data_dir + 'data_sintel_complete_test.csv',
                                  sep=',',
                                  columns=['image_path', 'label_path', 'invalid_path'],
                                  index=False,
                                  header=False)
            df_sintel_train_sample.to_csv(path_or_buf=data_dir + 'sample_data_sintel_complete_train.csv',
                                          sep=',',
                                          columns=['image_path', 'label_path', 'invalid_path'],
                                          index=False,
                                          header=False)
            df_sintel_valid_sample.to_csv(path_or_buf=data_dir + 'sample_data_sintel_complete_valid.csv',
                                          sep=',',
                                          columns=['image_path', 'label_path', 'invalid_path'],
                                          index=False,
                                          header=False)
            df_sintel_test_sample.to_csv(path_or_buf=data_dir + 'sample_data_sintel_complete_test.csv',
                                         sep=',',
                                         columns=['image_path', 'label_path', 'invalid_path'],
                                         index=False,
                                         header=False)

            # also save (unknown) test files:
            df_sintel_test_unknown = pd.DataFrame([[os.path.relpath(x, data_dir),
                                                    None,
                                                    None
                                                    ] for x in
                                                   glob.glob(data_dir_sintel_complete + 'test/clean/**/*.png')],
                                                  columns=['image_path', 'label_path', 'invalid_path'])

            df_sintel_test_unknown.to_csv(path_or_buf=data_dir + 'data_sintel_complete_test_unknown.csv', sep=',',
                                          columns=['image_path', 'label_path', 'invalid_path'], index=False,
                                          header=False)

    elif data_set == 'mpi_sintel_shading':

        # The problem is that the shading files (\*\*/out\_&ast;.png)
        # are named differently than the clean/albodo files
        # (\*\*/frame\_\*.png).
        # Also their numbering does not start with 1, 2, ...
        # Therefore we import each file path (clean, albedo and shading)
        # separately, sort it by the scene and frame (increasing) and
        # merge the three paths. Furthermore we have to get rid of some
        # scenes which are not included in either clean or albedo or shading.

        # maybe download data if necessary:
        download.maybe_download_and_extract(url=url_sintel_images,
                                            download_dir=data_dir + 'mpi-sintel-shading/')
        download.maybe_download_and_extract(url=url_sintel_albedo,
                                            download_dir=data_dir + 'mpi-sintel-shading/')
        download.maybe_download_and_extract(url=url_sintel_shading,
                                            download_dir=data_dir + 'mpi-sintel-shading/')

        if create_csv_lists:
            # import images and labels separateley:
            df_clean = pd.DataFrame(
                [os.path.relpath(x, data_dir) for x in
                 glob.glob(data_dir_sintel_shading + 'clean_noshadingtextures/**/*.png')],
                columns=['image_path'])
            df_clean[['scene', 'frame']] = df_clean['image_path'].apply(lambda row: row.split('/')[-2:]).apply(
                pd.Series)
            df_albedo = pd.DataFrame([os.path.relpath(x, data_dir) for x in
                                      glob.glob(data_dir_sintel_shading + 'albedo_noshadingtextures/**/*.png')],
                                     columns=['albedo_label_path'])
            df_albedo[['scene', 'frame']] = df_albedo['albedo_label_path'].apply(lambda row: row.split('/')[-2:]).apply(
                pd.Series)
            df_shading = pd.DataFrame(
                [os.path.relpath(x, data_dir) for x in glob.glob(data_dir_sintel_shading + 'shading/**/*.png')],
                columns=['shading_label_path'])
            df_shading[['scene', 'frame']] = df_shading['shading_label_path'].apply(
                lambda row: row.split('/')[-2:]).apply(pd.Series)
            try:
                df_invalid = pd.DataFrame(
                    [os.path.relpath(x, data_dir) for x in
                     glob.glob(data_dir_sintel_complete + 'training/invalid/**/*.png')],
                    columns=['invalid_path'])
                df_invalid[['scene', 'frame']] = df_invalid['invalid_path'].apply(
                    lambda row: row.split('/')[-2:]).apply(pd.Series)
            except KeyError:
                print('We need to download and extract the ' +
                      'mpi_sintel_complete dataset first to get the invalid ' +
                      'pixel mask.')
                main(data_set='mpi_sintel_complete',
                     data_dir=data_dir,
                     create_csv_lists=False)
                df_invalid = pd.DataFrame(
                    [os.path.relpath(x, data_dir) for x in
                     glob.glob(data_dir_sintel_complete + 'training/invalid/**/*.png')],
                    columns=['invalid_path'])
                df_invalid[['scene', 'frame']] = df_invalid['invalid_path'].apply(
                    lambda row: row.split('/')[-2:]).apply(pd.Series)

            # get list which contains scenes which have to be deleted:
            lst_del = [list(df_albedo[~df_albedo['scene'].isin(df_clean['scene'].unique())]['scene'].unique()) +
                       list(df_clean[~df_clean['scene'].isin(df_shading['scene'].unique())]['scene'].unique()) +
                       list(df_shading[~df_shading['scene'].isin(df_invalid['scene'].unique())]['scene'].unique()) +
                       list(df_invalid[~df_invalid['scene'].isin(df_albedo['scene'].unique())]['scene'].unique())
                       ][0]

            # delete scenes from lst_del, sort by ('scene', 'frame') and reset
            # index:
            df_clean = df_clean[~df_clean['scene'].isin(lst_del)]
            df_clean = df_clean.sort_values(['scene', 'frame'])
            df_clean.reset_index(drop=True, inplace=True)
            df_albedo = df_albedo[~df_albedo['scene'].isin(lst_del)]
            df_albedo = df_albedo.sort_values(['scene', 'frame'])
            df_albedo.reset_index(drop=True, inplace=True)
            df_shading = df_shading[~df_shading['scene'].isin(lst_del)]
            df_shading = df_shading.sort_values(['scene', 'frame'])
            df_shading.reset_index(drop=True, inplace=True)
            df_invalid = df_invalid[~df_invalid['scene'].isin(lst_del)]
            df_invalid = df_invalid.sort_values(['scene', 'frame'])
            df_invalid.reset_index(drop=True, inplace=True)

            # merge all four DataFrames and keep just important paths:
            df_merged = df_clean.merge(df_albedo, left_index=True,
                                       right_index=True, how='inner',
                                       suffixes=('_c', '_a'))
            df_merged = df_merged.merge(df_shading, left_index=True,
                                        right_index=True, how='inner',
                                        suffixes=('', '_s'))
            df_sintel2 = df_merged.merge(df_invalid, left_index=True,
                                         right_index=True, how='inner',
                                         suffixes=('_s', '_i'))[
                ['image_path', 'albedo_label_path', 'shading_label_path', 'invalid_path', 'scene_c']]

            df_sintel_train2, df_sintel_valid2, df_sintel_test2, df_sintel_train_sample2, \
                df_sintel_valid_sample2, df_sintel_test_sample2 = create_datasets_sintel(df_sintel2)

            # save complete data set, training data set, validation data set and
            # testing data set in separate data files:
            df_sintel2.to_csv(path_or_buf=data_dir + 'data_sintel_shading_complete.csv',
                              sep=',',
                              columns=['image_path', 'albedo_label_path', 'shading_label_path', 'invalid_path'],
                              index=False,
                              header=False)
            df_sintel_train2.to_csv(path_or_buf=data_dir + 'data_sintel_shading_train.csv',
                                    sep=',',
                                    columns=['image_path', 'albedo_label_path', 'shading_label_path', 'invalid_path'],
                                    index=False,
                                    header=False)
            df_sintel_valid2.to_csv(path_or_buf=data_dir + 'data_sintel_shading_valid.csv',
                                    sep=',',
                                    columns=['image_path', 'albedo_label_path', 'shading_label_path', 'invalid_path'],
                                    index=False,
                                    header=False)
            df_sintel_test2.to_csv(path_or_buf=data_dir + 'data_sintel_shading_test.csv',
                                   sep=',',
                                   columns=['image_path', 'albedo_label_path', 'shading_label_path', 'invalid_path'],
                                   index=False,
                                   header=False)
            df_sintel_train_sample2.to_csv(path_or_buf=data_dir + 'sample_data_sintel_shading_train.csv',
                                           sep=',',
                                           columns=['image_path', 'albedo_label_path', 'shading_label_path',
                                                    'invalid_path'],
                                           index=False,
                                           header=False)
            df_sintel_valid_sample2.to_csv(path_or_buf=data_dir + 'sample_data_sintel_shading_valid.csv',
                                           sep=',',
                                           columns=['image_path', 'albedo_label_path', 'shading_label_path',
                                                    'invalid_path'],
                                           index=False,
                                           header=False)
            df_sintel_test_sample2.to_csv(path_or_buf=data_dir + 'sample_data_sintel_shading_test.csv',
                                          sep=',',
                                          columns=['image_path', 'albedo_label_path', 'shading_label_path',
                                                   'invalid_path'],
                                          index=False,
                                          header=False)

    elif data_set == 'mit':
        # maybe download data if necessary:
        download.maybe_download_and_extract(url=url_mit, download_dir=data_dir)

        if create_csv_lists:
            df_mit = pd.DataFrame([[os.path.relpath(x, data_dir),
                                    os.path.relpath(x, data_dir).replace('original', 'reflectance'),
                                    os.path.relpath(x, data_dir).replace('original', 'shading')
                                    ] for x in glob.glob(data_dir_mit + '**/original.png')],
                                  columns=['image_path', 'albedo_label_path', 'shading_label_path'])

            # get training validation and testing data set of the mit data:
            df_mit_train, df_mit_valid, df_mit_test, _, _, \
            _ = create_datasets_mit(df=df_mit, p_train=0.8,
                                    p_valid=0.1, p_test=0.1, sample=False)

            # save complete data set, training data set, validation data set and
            # testing data set in separate data files:
            df_mit.to_csv(path_or_buf=data_dir + 'data_mit_complete.csv',
                          sep=',',
                          columns=['image_path', 'albedo_label_path',
                                   'shading_label_path'],
                          index=False,
                          header=False)
            df_mit_train.to_csv(path_or_buf=data_dir + 'data_mit_train.csv',
                                sep=',',
                                columns=['image_path', 'albedo_label_path',
                                         'shading_label_path'],
                                index=False,
                                header=False)
            df_mit_valid.to_csv(path_or_buf=data_dir + 'data_mit_valid.csv',
                                sep=',',
                                columns=['image_path', 'albedo_label_path',
                                         'shading_label_path'],
                                index=False,
                                header=False)
            df_mit_test.to_csv(path_or_buf=data_dir + 'data_mit_test.csv',
                               sep=',',
                               columns=['image_path', 'albedo_label_path',
                                        'shading_label_path'],
                               index=False,
                               header=False)
    else:
        raise ValueError("data_set must be in ['iiw', 'mpi_sintel_shading', " +
                         "'mpi_sintel_complete', 'mit']")


if __name__ == '__main__':
    # directory where to save csv files:
    data_dir = '/usr/udo/data/'
    main(data_set='iiw', data_dir=data_dir, create_csv_lists=True)
    #main(data_set='mpi_sintel_shading', data_dir=data_dir,
    #     create_csv_lists=True)
    #main(data_set='mpi_sintel_complete', data_dir=data_dir,
    #     create_csv_lists=True)
    #main(data_set='mit', data_dir=data_dir, create_csv_lists=True)

