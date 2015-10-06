#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Parsing the garment datasets

    This script will extract datasets from MySQL based on
    the attributes that has configured.
"""
import config as cfg
import glob
import sys
import numpy as np
import lmdb
import lmdb_util as lu
import augment_util as au
import cv2 as cv
import os

from random import shuffle
from attributes import parsing_attribute_configfile
from attributes import get_attribute_index
from datamodel import GarmentDataModel


def get_item_entries(data_model, attributes, attribute_indices):
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
        3. attribute_indices    | Dictionary

        OUTPUT:
        1. ori_training_items   | List                   | List that contains original training entry
        2. ori_validate_list    | List                   | List that contains original validation entry
        3. ori_test_list        | List                   | List that contains original test entry

    """

    # Variables
    entry_lists = []

    # Iterate the attribute to build list
    for count, attribute_entry in enumerate(attributes):

        # Extract variables from attribute entry
        lmdb_grop_index = attribute_entry['lmdb_group']
        component_name = attribute_entry['component']
        attribute_name = attribute_entry['attribute']
        attribute_label = int(attribute_entry['attri_label'])
        sql_query = attribute_entry['query_string']
        bbox_points = int(attribute_entry['bbox_pts'])
        minimal_width_ratio = float(attribute_entry['ratio'])

        # Extra info for pre-processing on bounding box
        offset_x = float(attribute_entry['x_offset'])
        offset_y = float(attribute_entry['y_offset'])
        x_ext_factor = float(attribute_entry['x_ext_factor'])
        y_ext_factor = float(attribute_entry['y_ext_factor'])

        # Query the raw data from MySQL database
        raw_data = data_model.query(sql_query)

        for data in raw_data:

            # Extract image name
            image_name = data[0]

            # Extract bounding bounding boxes
            start_pt = np.asarray([sys.float_info.max, sys.float_info.max])
            end_pt = np.asarray([sys.float_info.min, sys.float_info.min])

            for n in range(0, bbox_points):
                x = data[2*n + 1]
                y = data[2*n + 2]

                # Find the max and min for end and start point
                if x < start_pt[0]:
                    start_pt[0] = int(x)

                if x >= end_pt[0]:
                    end_pt[0] = int(x)

                if y < start_pt[1]:
                    start_pt[1] = int(y)

                if y >= end_pt[1]:
                    end_pt[1] = int(y)

            # Add image name and bounding box to entry
            entry = (image_name, attribute_label, start_pt, end_pt, (offset_x, offset_y), (x_ext_factor, y_ext_factor),
                     minimal_width_ratio, lmdb_grop_index)

            entry_lists.append(entry)

    # If we need clamp the datasets:
    if cfg.maximal_total_images != -1:
        entry_lists = entry_lists[:cfg.maximal_total_images]

    # Add negative images
    neg_images = glob.glob(cfg.neg_image_directory + '*.jpg')
    neg_images += glob.glob(cfg.neg_image_directory + '*.png')

    for neg_img in neg_images[:cfg.max_neg_img_count]:
        start_pt_x = np.random.randint(0, 350)
        start_pt_y = np.random.randint(0, 200)

        bbox_base_size = np.random.randint(40, 140)
        bbox_size_x, bbox_size_y = 0, 0

        # Clamp the size
        bbox_size_x = 479 - start_pt[0] if start_pt_x + bbox_base_size >= 480 else bbox_base_size
        bbox_size_y = 299 - start_pt[1] if start_pt_y + bbox_base_size >= 300 else bbox_base_size

        start_pt = np.asarray([start_pt_x, start_pt_y])
        end_pt = np.asarray([bbox_size_x + start_pt[0], bbox_size_y + start_pt[1]])

        # Add to entry list
        entry = (neg_img, 0, start_pt, end_pt, (0, 0), (1.0, 1.0), 1.0)
        entry_lists.append(entry)

        # if cfg.debug_gen_flag is True:
        #     img = cv.imread(neg_img, cv.IMREAD_COLOR)
        #     img = cv.resize(img, (480, 300))
        #     cv.rectangle(img, (start_pt[0], start_pt[1]), (end_pt[0], end_pt[1]), (0, 255, 0), 1)
        #     cv.imshow('debug_preview', img)
        #     cv.waitKey(0)

    # Shuffle the 'entry_list'
    shuffle(entry_lists)

    # Extract training, validation, test entry lists
    train_size = int(len(entry_lists) * cfg.train_imgs_ratio)
    ori_training_items = entry_lists[0: train_size]
    offset = train_size

    validate_size = int(len(entry_lists) * cfg.valid_imgs_ratio)
    ori_validate_items = entry_lists[offset: offset + validate_size]
    offset += validate_size

    test_size = offset + int(len(entry_lists) * cfg.test_imgs_ratio)
    if test_size > len(entry_lists):
        test_size = len(entry_lists) - 1

    ori_test_items = entry_lists[offset: test_size]

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

    for count, basic_entry in enumerate(training_list):
        # Generate an array that contains flag determing whether an image need to
        # be flipped
        flip_flags = np.random.rand(cfg.augment_size) > (1 - cfg.flip_percent)

        # Generate translation and rotation magnitude
        trans_magnitudes = None
        zooming_magnitudes = None
        if cfg.distribution_method is 'Gaussian':
            trans_magnitudes = cfg.translate_factor * \
                              np.clip(np.random.normal(0, gaussian_sigma, (cfg.augment_size, 2)), -1, 1)

            zooming_magnitudes = cfg.zooming_factor * \
                                 np.clip(np.random.normal(0, gaussian_sigma, cfg.augment_size), -1, 1)

        elif cfg.distribution_method is 'Uniform':
            trans_magnitudes = cfg.translate_factor * np.random.uniform(-1, 1, (cfg.augment_size, 2))
            zooming_magnitudes = cfg.zooming_factor * np.random.uniform(-1, 1, cfg.augment_size)

        # Add to the augmentation list with corresponding augmentation parameters
        # information in 'basic_entry' as well as the augmentation parameters will added to a new list
        for i in range(0, cfg.augment_size):
            aug_training_list.append((basic_entry, flip_flags[i], trans_magnitudes[i], zooming_magnitudes[i]))

        # Print info
        print 'augmenting: ', count

    # Shuffle the training list
    shuffle(aug_training_list)

    return aug_training_list


def generate_lmdbs(list, img_lmdb_path, bbox_lmdb_path, attributes, aug_flag=False):
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

    # Initialize the attribute lmdbs
    attri_lmdbs = []
    for i in range(0, cfg.attri_lmdb_groups):
        lu.del_and_create(cfg.lmdb_output_path + 'attri_' + i + '.lmdb')
        attri_lmdb_env = lmdb.Environment(bbox_lmdb_path, map_size=1099511627776)
        attri_lmdb_txn = bbox_env.begin(write=True, buffers=True)
        attri_lmdbs.append((attri_lmdb_env, attri_lmdb_txn))

    if cfg.debug_gen_flag is False:
        img_env = lmdb.Environment(img_lmdb_path, map_size=1099511627776)
        img_txn = img_env.begin(write=True, buffers=True)

        bbox_env = lmdb.Environment(bbox_lmdb_path, map_size=1099511627776)
        bbox_txn = bbox_env.begin(write=True, buffers=True)

    # Initialize random keys
    keys = np.arange(len(list))
    np.random.shuffle(keys)
    fail_count = 0

    for count, entry in enumerate(list):

        # Print info every 100 iterations
        if count % 100 == 0:
            print 'Currently processing on %d' % count

        # Extract infomations
        basic_info = entry if aug_flag is False else entry[0]
        image_name = basic_info[0]
        attri_label = int(basic_info[1])
        bbox = np.asarray((basic_info[2], basic_info[3])).flatten()
        offset_factor = np.asarray(basic_info[4]).flatten()
        ext_factor = np.asarray(basic_info[5]).flatten()
        minimal_width_ratio = basic_info[6]
        lmdb_group_index = int(basic_info[7])

        flip_flag = False if aug_flag is True else bool(entry[1])
        trans_magnitude = (0, 0) if aug_flag is False else entry[2]
        zooming_magnitude = 0 if aug_flag is False else entry[3]

        # Read image
        img, img_size = None, None
        if attri_label is not 0:
            img_file_path = cfg.image_directory + image_name + '.jpg'
            if not os.path.exists(img_file_path):
                continue

            img = cv.imread(img_file_path, cv.IMREAD_COLOR)
            img_height, img_width, img_channels = img.shape
            img_size = np.asarray((img_width, img_height)).flatten()
        else:   # Negative images
            img_file_path = image_name
            if not os.path.exists(img_file_path):
                continue

            img = cv.imread(img_file_path, cv.IMREAD_COLOR)
            img = cv.resize(img, (480, 300))
            img_size = np.asarray((480, 300)).flatten()

        offset = (offset_factor[0] * img_size[0], offset_factor[1] * img_size[1])

        # Add basic offset
        bbox = au.translate_box(bbox, img_size, offset)
        # If need flip
        if flip_flag is True:
            img = cv.flip(img, 1)
            bbox = (img_size[0] - bbox[2], bbox[1],
                    img_size[0] - bbox[0], bbox[3])

        # Extend the crop region with random zooming
        ext_bbox = au.extend_bbox_xy(bbox, img_size,
                                     ext_factor[0] * (1 + zooming_magnitude), ext_factor[1] * (1 + zooming_magnitude),
                                     xy_ratio=minimal_width_ratio)

        ext_bbox_size = (ext_bbox[2] - ext_bbox[0], ext_bbox[3] - ext_bbox[1])

        # Apply random translate
        random_offset = (trans_magnitude[0] * ext_bbox_size[0], trans_magnitude[1] * ext_bbox_size[1])
        ext_bbox = au.translate_box(ext_bbox, img_size, random_offset)

        if cfg.debug_gen_flag is True:
            # Draw bounding box in image
            cv.rectangle(img, (int(ext_bbox[0]), int(ext_bbox[1])),
                                    (int(ext_bbox[2]), int(ext_bbox[3])),
                                    (255, 0, 0), 2)

            if cfg.enable_debug_gen_file is True:
                # Write preview image to file
                cv.imwrite(cfg.debug_gen_path + image_name + '.jpg', img)
            else:
                # Show in window
                print 'Current:' + image_name + ' Label:' + str(attri_label)
                cv.imshow('debug_preview', img)
                cv.waitKey(0)
            continue

        # Check the bbox
        if ext_bbox[3] < ext_bbox[1] or ext_bbox[2] < ext_bbox[0]:
            fail_count += 1
            continue

        # Crop the image
        crop_img = img[int(ext_bbox[1]): int(ext_bbox[3]), int(ext_bbox[0]): int(ext_bbox[2])]
        if crop_img is None or img is None:
            fail_count += 1
            continue

        # Resize image
        crop_img = cv.resize(crop_img, cfg.output_img_size)

        # Preview the crop_img if needed
        if cfg.debug_gen_flag is True:
            cv.imshow('debug_preview', crop_img)
            cv.waitKey(0)
            continue

        # Push to lmdb
        train_img_datum = lu.generate_img_datum(crop_img, label=attri_label)
        bbox_datum = lu.generate_array_datum(ext_bbox, is_float=True)

        # Put into attribute lmdb, not that the size of a lmdb is N+1, 1 for Background


        # Write to lmdb every 10000 times
        key = '%010d' % keys[count]
        img_txn.put(key, train_img_datum.SerializeToString())
        bbox_txn.put(key, bbox_datum.SerializeToString())

        if count % 10000 == 0:
            # Write to lmdb every 10000 times
            img_txn.commit()
            bbox_txn.commit()
            img_txn = img_env.begin(write=True, buffers=True)
            bbox_txn = bbox_env.begin(write=True, buffers=True)

    if cfg.debug_gen_flag is False:
        # Close the lmdb
        img_txn.commit()
        bbox_txn.commit()
        img_env.close()
        bbox_env.close()

        for i in range(0, cfg.attri_lmdb_groups):
            attri_lmdbs[i][1].commit()
            attri_lmdbs[i][0].close()

        print 'Push items into lmdb has finished.'

    print 'Total: ' + str(len(list)) + ' Failures: ' + str(fail_count)


def generate_csvfile(list, csvfile_path, aug_flag=False):

    """
        Generate csv file based on entry list

        Example of CSV file format as follows:
            image_file,  label (in number), bbox_x1, bbox_y1, bbox_x2, bbox_y2, flip_flag,
            20.jpg    ,                  1,                        22,21,23,42,      True,

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

    for count, entry in enumerate(list):

        # Print info every 100 iterations
        if count % 100 == 0:
            print 'Currently processing on %d\n' % count

        # Extract infomations
        basic_info = entry if aug_flag is False else entry[0]
        image_name = basic_info[0]
        attri_label = basic_info[1]
        bbox = np.asarray((basic_info[2], basic_info[3])).flatten()
        offset_factor = np.asarray(basic_info[4]).flatten()
        ext_factor = np.asarray(basic_info[5]).flatten()
        minimal_width_ratio = basic_info[6]
        lmdb_group_index = basic_info[7]

        flip_flag = False if aug_flag is True else bool(entry[1])
        trans_magnitude = (0, 0) if aug_flag is False else entry[2]
        zooming_magnitude = 0 if aug_flag is False else entry[3]

        # Read image
        img, img_size = None, None
        if attri_label is not 0:
            img_file_path = cfg.image_directory + image_name + '.jpg'
            if not os.path.exists(img_file_path):
                continue

            img = cv.imread(img_file_path, cv.IMREAD_COLOR)
            img_height, img_width, img_channels = img.shape
            img_size = np.asarray((img_width, img_height)).flatten()
        else:   # Negative images
            img_file_path = image_name
            if not os.path.exists(img_file_path):
                continue

            img = cv.imread(img_file_path, cv.IMREAD_COLOR)
            img_size = np.asarray((320, 200)).flatten()

        offset = (offset_factor[0] * img_size[0], offset_factor[1] * img_size[1])

        # Add basic offset
        bbox = au.translate_box(bbox, img_size, offset)
        # If need flip
        if flip_flag is True:
            img = cv.flip(img, 1)
            bbox = (img_size[0] - bbox[2], bbox[1],
                    img_size[0] - bbox[0], bbox[3])

        # Extend the crop region with random zooming
        ext_bbox = au.extend_bbox_xy(bbox, img_size,
                                     ext_factor[0] * (1 + zooming_magnitude), ext_factor[1] * (1 + zooming_magnitude),
                                     xy_ratio=minimal_width_ratio)

        ext_bbox_size = (ext_bbox[2] - ext_bbox[0], ext_bbox[3] - ext_bbox[1])

        # Apply random translate
        random_offset = (trans_magnitude[0] * ext_bbox_size[0], trans_magnitude[1] * ext_bbox_size[1])
        ext_bbox = au.translate_box(ext_bbox, img_size, random_offset)

        if cfg.debug_gen_flag is True:

            # Draw bounding box in image
            cv.rectangle(img, (int(ext_bbox[0]), int(ext_bbox[1])),
                                    (int(ext_bbox[2]), int(ext_bbox[3])),
                                    (255, 0, 0), 2)

            if cfg.enable_debug_gen_file is True:
                # Write preview image to file
                cv.imwrite(cfg.debug_gen_path + image_name + '.jpg', img)
            else:
                # Show in window
                print 'Current:' + image_name + ' Label:' + str(attri_label)
                cv.imshow('debug_preview', img)
                cv.waitKey(0)

            continue

        # Write to csv file
        csv_file.write('%s, %d, %f, %f, %f, %f, %d,\n' % (image_name, attri_label,
                                                          ext_bbox[0], ext_bbox[1], ext_bbox[2], ext_bbox[3],
                                                          flip_flag))

    csv_file.close()

    if cfg.debug_gen_flag is False:
        print 'Push items into csv file has finished, \nTotal: %d\n' % len(list)


if __name__ == '__main__':
    """
        [TEST] preview window initialization
    """
    if cfg.debug_gen_flag is True:
        cv.namedWindow('debug_preview')
        # Check if we need generate debug preview files, we need to create
        # directory
        if cfg.enable_debug_gen_file is True and os.path.exists(cfg.debug_gen_path) is False:
            os.mkdir(cfg.debug_gen_path)

    """
        Main pipeline for generate lmdbs
    """
    # Add PyCaffe to sys.path
    sys.path.append(cfg.pycaffe_path)

    # Create data provider
    data_model = GarmentDataModel(cfg.mysql_hostname,
                                  cfg.mysql_username,
                                  cfg.mysql_userpasswd,
                                  cfg.mysql_dbname)

    # Read attributes configuration from file, the configuration
    # is a dictionary with multiple value fields
    attributes = parsing_attribute_configfile(cfg.attribute_file_path)

    attri_indices = get_attribute_index(cfg.attri_index_file_path)

    # Parsing the data into training, validation, test list
    # ori_training_list, ori_validate_list, ori_test_list = get_item_entries(data_model, attributes)

    # Augment the training dataset
    aug_training_list = augment_training_data(ori_training_list)

    # Generate lmdbs
    print 'Start to create training lmdbs.'
    generate_lmdbs(aug_training_list, cfg.train_img_dbpath, cfg.train_bbox_dbpath, aug_flag=True)

    print 'Start to create validation lmdbs.'
    generate_lmdbs(ori_validate_list, cfg.valid_img_dbpath, cfg.valid_bbox_dbpath, aug_flag=False)

    # Generate test lists
    generate_csvfile(ori_test_list, cfg.test_list_path)

    # Destory window if needed
    if cfg.debug_gen_flag is True:
        cv.destroyWindow('debug_preview')