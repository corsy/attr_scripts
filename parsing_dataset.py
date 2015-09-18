#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Parsing the garment datasets

    This script will extract datasets from MySQL based on
    the attributes that has configured.
"""
import config as cfg
from attributes import parsing_attribute_configfile
from datamodel import GarmentDataModel


def get_item_entries(data_model, attributes):

    """
        Get the item entries for training, validation, test

        Entries are collected in three list, note that this list
        contains no entry belongs to dataset augmentation, in order
        to augment the training dataset, function '' need to be
        invoked.

        INPUT:
           PARAMS          | TYPE                   | DESCRIPTION
        1. data_model      | GarmentDataModel       | Providing garment data model by executing MySQL query
        2. attributes      | List                   | Contains attributes details restored in a dictionary
                                                      (e.g. component name, attributes, query to be executed)
        OUTPUT:
        1. training_list   | List                   | List that contains original training entry
        2. validate_list   | List                   | List that contains original validation entry
        3. test_list       | List                   | List that contains original test entry

    """

    # Variables
    ori_training_items = []
    ori_validate_items = []
    ori_test_items = []

    # Iterate the attribute to build list
    for i, attribute_entry in attributes:

    return 0

def augment_training_data(training_list):

    """
        Augment the training data

        This function will augment the training data base on the configurations detailed
        in 'config.py'. Augmentation contains flip, random tranlsation,

        INPUT:
           PARAMS            | TYPE                   | DESCRIPTION
        1. training_list     | GarmentDataModel       | List that contains original training entry

        OUTPUT:
           PARAMS            | TYPE                   | DESCRIPTION
        1. aug_training_list | List                   | List that contains augmented training entry

    """

def generate_lmdbs(list, img_lmdb_path, bbox_lmdb_path):

    """
        Generate lmdb database based on entry list

        INPUT:
           PARAMS            | TYPE                   | DESCRIPTION
        1. list              | List                   | List that contains original entry, list could be training,
                                                        validation, test set.
        2. img_lmdb_path     | String                 | Output path for image lmdb.
        3. bbox_lmdb_path    | String                 | Output path for bounding box lmdb.

    """

def generate_csvfile(list, csvfile_path):

    """
        Generate csv file based on entry list

        CSV file format as follows:
            image_file, bbox pts(x1;y1;x2;y2), label (in number) ,
            20.jpg    ,     22;21;23;42      , 1                 ,

        INPUT:
           PARAMS            | TYPE                   | DESCRIPTION
        1. list              | List                   | List that contains original entry, list could be training,
                                                        validation, test set.
        2. img_list_path     | String                 | Output path for image set list.
        3. bbox_list_path    | String                 | Output path for bounding box set list.

    """

if __name__ == '__main__':

    """
        Main pipeline for generate lmdbs
    """
    # Create data provider
    data_model = GarmentDataModel(cfg.mysql_hostname,
                                  cfg.mysql_username,
                                  cfg.mysql_userpasswd,
                                  cfg.mysql_dbname)

    # Read attributes configuration from file, the configuration
    # is a dictionary with multiple value fields
    attributes = parsing_attribute_configfile(cfg.attribute_file_path)

    # Parsing the data into training, validation, test list
    ori_training_list, ori_validate_list, ori_test_list = get_item_entries(data_model, attributes)

    # Augment the training dataset
    # TODO(Luwei): implement 'augment_training_data' function
    # aug_training_list = augment_training_data(ori_test_list)

    # Generate lmdbs
    generate_lmdbs(ori_training_list, cfg.train_img_dbpath, cfg.train_bbox_dbpath)
    generate_lmdbs(ori_validate_list, cfg.valid_img_dbpath, cfg.valid_bbox_dbpath)

    # Generate test lists
    generate_csvfile(ori_test_list, cfg.test_list_path)

