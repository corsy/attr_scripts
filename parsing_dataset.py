#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Parsing the garment datasets

    This script will extract datasets from MySQL based on
    the attributes that has configured.
"""
import config as cfg
import sys
import numpy as np
import lmdb
import lmdb_util as lu
import augment_util as au
import cv2 as cv

from random import shuffle
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
           PARAMS               | TYPE                   | DESCRIPTION
        1. data_model           | GarmentDataModel       | Providing garment data model by executing MySQL query
        2. attributes           | List                   | Contains attributes details restored in a dictionary
                                                      (e.g. component name, attributes, query to be executed)
        OUTPUT:
        1. ori_training_items   | List                   | List that contains original training entry
        2. ori_validate_list    | List                   | List that contains original validation entry
        3. ori_test_list        | List                   | List that contains original test entry

    """

    # Variables
    entry_lists = []

    # Iterate the attribute to build list
    for i, attribute_entry in attributes:

        # Extract variables from attribute entry
        component_name = attribute_entry['component']
        attribute_name = attribute_entry['attribute']
        attribute_label = i
        sql_query = attribute_entry['query']
        bbox_points = int(attribute_entry['bbox_pts'])

        # Query the raw data from MySQL database
        raw_data = data_model.query(sql_query)

        # Extract image name
        image_name = raw_data[0]

        # Extract bounding bounding boxes
        start_pt = (sys.float_info.max, sys.float_info.max)
        end_pt = (sys.float_info.min, sys.float_info.min)

        for n in range(0, bbox_points):
            x = raw_data[2*n + 1]
            y = raw_data[2*n + 2]

            # Find the max and min for end and start point
            if x < start_pt[0]:
                start_pt[0] = x
            elif x >= end_pt[0]:
                end_pt[0] = x

            if y < start_pt[1]:
                start_pt[1] = y
            elif y >= end_pt[1]:
                end_pt[1] = y

        # Add image name and bounding box to entry
        entry = (image_name, attribute_label, start_pt, end_pt)
        entry_lists.append(entry)

    # Shuffle the 'entry_list'
    shuffle(entry_lists)

    # Extract training, validation, test entry lists
    ori_training_items = entry_lists[0: cfg.train_size]
    offset = cfg.train_size
    ori_validate_items = entry_lists[offset: offset + cfg.validation_size]
    offset += cfg.validation_size
    ori_test_items = entry_lists[offset: cfg.test_size]

    return ori_training_items, ori_validate_items, ori_test_items

def augment_training_data(training_list):
    """
        Augment the training data

        This function will augment the training data base on the configurations detailed
        in 'config.py'. Augmentation contains flip, random translation,

        INPUT:
           PARAMS            | TYPE                   | DESCRIPTION
        1. training_list     | GarmentDataModel       | List that contains original training entry

        OUTPUT:
           PARAMS            | TYPE                   | DESCRIPTION
        1. aug_training_list | List                   | List that contains augmented training entry

    """
    aug_training_list = []

    # Iterate the training_list to get 'basic_entry'
    # 'basic_entry' only contains image name, attribute label,
    gaussian_sigma = 0.4

    for count, basic_entry in training_list:

        # Generate an array that contains flag determing whether an image need to
        # be flipped
        filp_flags = np.random.rand(cfg.augment_size) > (1 - cfg.flip_percent)

        # Generate translation and rotation magnitude
        trans_magnitudes = None
        zooming_magnitudes = None
        if cfg.distribution_method is 'Gaussian':
            trans_magnitudes = cfg.translate_factor * \
                              np.clip(np.random.normal(0, gaussian_sigma, (cfg.augment_size, 2)), -1, 1)

            zooming_magnitudes = cfg.zooming_factor * \
                                np.clip(np.random.normal(0, gaussian_sigma, (cfg.augment_size, 2)), -1, 1)

        elif cfg.distribution_method is 'Uniform':
            trans_magnitudes = cfg.translate_factor * np.random.uniform(-1, 1, (cfg.augment_size, 2))
            zooming_magnitudes = cfg.zooming_factor * np.random.uniform(-1, 1, cfg.augment_size)

        # Add to the augmentation list with corresponding augmentation parameters
        # information in 'basic_entry' as well as the augmentation parameters will added to a new list
        for i in range(0, cfg.augment_size):
            aug_training_list.append((basic_entry, filp_flags[i], trans_magnitudes[i], zooming_magnitudes[i]))

        # Print info
        print 'augmenting: ', count

    # Shuffle the training list
    shuffle(aug_training_list)

    return aug_training_list

def generate_lmdbs(list, img_lmdb_path, bbox_lmdb_path, aug_flag=False):

    """
        Generate lmdb database based on entry list

        INPUT:
           PARAMS            | TYPE                   | DESCRIPTION
        1. list              | List                   | List that contains original entry, list could be training,
                                                        validation, test set. List should contains 'basic_entry' that
                                                        contains image_name, bounding box, attribute label as well as
                                                        flip flag, magnitude of random translation and zooming.

        2. img_lmdb_path     | String                 | Output path for image lmdb.
        3. bbox_lmdb_path    | String                 | Output path for bounding box lmdb.
        4. aug_flag          | bool                   | Mark if we need to augment the datasets [defalut: False]

    """
    # Initialize the lmdbs
    lu.del_and_create(img_lmdb_path)
    lu.del_and_create(bbox_lmdb_path)

    img_env, img_txn, bbox_env, bbox_txn = None, None, None, None

    if cfg.debug_gen_flag is True:
        img_env = lmdb.Environment(img_lmdb_path, map_size=1099511627776)
        img_txn = img_env.begin(write=True, buffers=True)

        bbox_env = lmdb.Environment(bbox_lmdb_path, map_size=1099511627776)
        bbox_txn = bbox_env.begin(write=True, buffers=True)

    # Initialize random keys
    keys = np.arange(len(list))
    np.random.shuffle(keys)

    for count, entry in list:

        # Print info every 100 iterations
        if count % 100 == 0:
            print 'Currently processing on %d\n' % count

        # Extract infomations
        basic_info = entry[0]
        image_name = basic_info[0]
        attri_label = basic_info[1]
        bbox = (basic_info[2], basic_info[3])

        flip_flag = False
        trans_magnitude = (0, 0)
        zooming_magnitude = 0

        if aug_flag is True:
            flip_flag = entry[1]
            trans_magnitude = entry[2]
            zooming_magnitude = entry[3]

        # Read image
        img = cv.imread(cfg.image_directory + image_name, cv.IMREAD_COLOR)
        img_size = np.asarray(img.shape).flatten()

        # Extend the crop region with random zooming
        ext_bbox = au.extend_bbox(bbox, img_size, cfg.base_bbox_zooming_factor * (1 + zooming_magnitude))
        ext_bbox_size = (ext_bbox[2] - ext_bbox[0], ext_bbox[3] - ext_bbox[1])

        # If need flip
        if flip_flag is True:
            img = cv.flip(img, 1)
            ext_bbox[0] = img_size[0] - ext_bbox[0]
            ext_bbox[2] = img_size[0] - ext_bbox[2]

        # Apply random translate
        offset = (trans_magnitude[0] * ext_bbox_size[0], trans_magnitude[1] * ext_bbox_size[1])
        ext_bbox = au.translate_box(ext_bbox, img_size, offset)

        if cfg.debug_gen_flag is True:

            # Draw bounding box in image
            img = cv.rectangle(img, (ext_bbox[0], ext_bbox[1]), (ext_bbox[2], ext_bbox[3]), (255, 0, 0), 2)

            if cfg.enable_debug_gen_file is True:
                # Write preview image to file
                cv.imwrite(cfg.debug_gen_path + image_name + '.jpg', img)
            else:
                # Show in window
                cv.imshow('preview', img)
                cv.waitKey(0)

            continue

        # Push to lmdb
        train_img_datum = lu.generate_img_datum(img, label=attri_label)
        bbox_datum = lu.generate_array_datum(ext_bbox)

        key = '%010d' % keys[count]
        img_txn.put(key, train_img_datum.SerializeToString())
        bbox_txn.put(key, bbox_datum.SerializeToString())

        if count % 10000 == 0:
            img_txn.commit()
            bbox_txn.commit()
            img_txn = img_env.begin(write=True, buffers=True)
            bbox_txn = bbox_env.begin(write=True, buffers=True)

    if cfg.debug_gen_flag is False:
        img_txn.commit()
        bbox_txn.commit()
        img_env.close()
        bbox_env.close()
        print 'Push items into lmdb has finished, \nTotal: %d\n' % len(list)

def generate_csvfile(list, csvfile_path, aug_flag=False):

    """
        Generate csv file based on entry list

        Example of CSV file format as follows:
            image_file,  label (in number), bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            20.jpg    ,                  1,                        22,21,23,42,

        INPUT:
           PARAMS            | TYPE                   | DESCRIPTION
        1. list              | List                   | List that contains original entry, list could be training,
                                                        validation, test set.
        2. img_list_path     | String                 | Output path for image set list.
        3. bbox_list_path    | String                 | Output path for bounding box set list.
        4. aug_flag          | bool                   | Mark if we need to augment the datasets [defalut: False]

    """

    # Initialize the lmdbs
    csv_file = open(csvfile_path, 'w')

    csv_file.write('image_file, label, bbox_x1, bbox_y1, bbox_x2, bbox_y2\n')

    for count, entry in list:

        # Print info every 100 iterations
        if count % 100 == 0:
            print 'Currently processing on %d\n' % count

        # Extract infomations
        basic_info = entry[0]
        image_name = basic_info[0]
        attri_label = basic_info[1]
        bbox = (basic_info[2], basic_info[3])

        flip_flag = False
        trans_magnitude = (0, 0)
        zooming_magnitude = 0

        if aug_flag is True:
            flip_flag = entry[1]
            trans_magnitude = entry[2]
            zooming_magnitude = entry[3]

        # Read image
        img = cv.imread(cfg.image_directory + image_name, cv.IMREAD_COLOR)
        img_size = np.asarray(img.shape).flatten()

        # Extend the crop region with random zooming
        ext_bbox = au.extend_bbox(bbox, img_size, cfg.base_bbox_zooming_factor * (1 + zooming_magnitude))
        ext_bbox_size = (ext_bbox[2] - ext_bbox[0], ext_bbox[3] - ext_bbox[1])

        # If need flip
        if flip_flag is True:
            img = cv.flip(img, 1)
            ext_bbox[0] = img_size[0] - ext_bbox[0]
            ext_bbox[2] = img_size[0] - ext_bbox[2]

        # Apply random translate
        offset = (trans_magnitude[0] * ext_bbox_size[0], trans_magnitude[1] * ext_bbox_size[1])
        ext_bbox = au.translate_box(ext_bbox, img_size, offset)

        if cfg.debug_gen_flag is True:

            # Draw bounding box in image
            img = cv.rectangle(img, (ext_bbox[0], ext_bbox[1]), (ext_bbox[2], ext_bbox[3]), (255, 0, 0), 2)

            if cfg.enable_debug_gen_file is True:
                # Write preview image to file
                cv.imwrite(cfg.debug_gen_path + image_name + '.jpg', img)
            else:
                # Show in window
                cv.imshow('debug_preview', img)
                cv.waitKey(0)

            continue

        # Write to csv file
        csv_file.write('%s, %d, %f, %f, %f, %f,\n' % (image_name, attri_label,
                                                     ext_bbox[0], ext_bbox[1], ext_bbox[2]), ext_bbox[3])

    csv_file.close()

    if cfg.debug_gen_flag is False:
        print 'Push items into csv file has finished, \nTotal: %d\n' % len(list)


if __name__ == '__main__':
    """
        [TEST] preview window initialization
    """
    if cfg.debug_gen_flag is True:
        cv.namedWindow('debug_preview')

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
    aug_training_list = augment_training_data(ori_test_list)

    # Generate lmdbs
    generate_lmdbs(ori_training_list, cfg.train_img_dbpath, cfg.train_bbox_dbpath, aug_flag=True)
    generate_lmdbs(ori_validate_list, cfg.valid_img_dbpath, cfg.valid_bbox_dbpath, aug_flag=False)

    # Generate test lists
    generate_csvfile(ori_test_list, cfg.test_list_path)

    # Destory window if needed
    if cfg.debug_gen_flag is True:
        cv.destroyAllWindows()