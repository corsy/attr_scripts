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
import matplotlib.pyplot as plt
import os

from random import shuffle
from attributes import parsing_attribute_configfile
from attributes import get_attribute_index
from datamodel import GarmentDataModel

# For debug
from bar_test import BarPlotor
bar_plt = BarPlotor()
img_fig = plt.figure('Preview Image')
if cfg.debug_gen_flag is True:
    bar_fig = plt.figure('Bar debug')

def show_img(image, fig):
    fig.figimage(image, 0, 0)
    return fig

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
    bar_plt = BarPlotor()

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


def generate_lmdbs(list, img_lmdb_path, bbox_lmdb_path, lmdb_prefix, aug_flag=False):
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
    # Check output path is exists
    if not os.path.exists(cfg.lmdb_output_path):
        os.mkdir(cfg.lmdb_output_path)

    # Initialize the lmdbs
    lu.del_and_create(img_lmdb_path)
    lu.del_and_create(bbox_lmdb_path)

    img_env, img_txn, bbox_env, bbox_txn = None, None, None, None
    attri_lmdbs = []

    # Initialize the lmdbs
    if cfg.debug_gen_flag is False:
        img_env = lmdb.Environment(img_lmdb_path, map_size=1099511627776)
        img_txn = img_env.begin(write=True, buffers=True)

        bbox_env = lmdb.Environment(bbox_lmdb_path, map_size=1099511627776)
        bbox_txn = bbox_env.begin(write=True, buffers=True)

        for i in range(0, len(cfg.attributes_index)):
            lmdb_path = cfg.lmdb_output_path + 'group_' + str(i) + '_' + lmdb_prefix +'.lmdb'
            lu.del_and_create(lmdb_path)
            attri_lmdb_env = lmdb.Environment(lmdb_path, map_size=1099511627776)
            attri_lmdb_txn = attri_lmdb_env.begin(write=True, buffers=True)
            attri_lmdbs.append([attri_lmdb_env, attri_lmdb_txn])

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
        attri_labels = basic_info[1]
        bbox = np.asarray((basic_info[2])).flatten()
        offset_factor = np.asarray(basic_info[3], dtype=float).flatten()
        ext_factor = np.asarray(basic_info[4], dtype=float).flatten()
        minimal_width_ratio = float(basic_info[5])
        comp_idx = int(basic_info[6])

        flip_flag = False if aug_flag is True else bool(entry[1])
        trans_magnitude = (0, 0) if aug_flag is False else entry[2]
        zooming_magnitude = 0 if aug_flag is False else entry[3]

        # Create attribute label matrix
        attri_idx_groups = []
        for group in range(0, len(cfg.attributes_index)):
            group_len = cfg.attributes_index[group]['count']
            labels = 0      # Default index: Background
            attri_idx_groups.append(labels)

        # Modify attri_idx_groups
        if attri_labels is not None:
            for attribute in attri_labels:
                group_idx = attribute['group_idx']
                attri_idx = attribute['attri_idx']
                group_len = cfg.attributes_index[group_idx]['count']
                new_labels = attri_idx
                attri_idx_groups[group_idx] = new_labels

        # Read image
        img, img_size = None, None
        if attri_labels is not None:
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
                print 'Current:' + image_name
                cv.imshow('debug_preview', img)
                bar_plt.show_label_bar(attri_idx_groups, bar_fig)
                cv.imshow('bar_preview', cv.imread('test.png'))
                cv.waitKey(0)
                # pass
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
        train_img_datum = lu.generate_img_datum(crop_img, label=comp_idx)
        bbox_datum = lu.generate_array_datum(ext_bbox, is_float=True)

        # Write to lmdb every 10000 times
        key = '%010d' % keys[count]
        img_txn.put(key, train_img_datum.SerializeToString())
        bbox_txn.put(key, bbox_datum.SerializeToString())

        # Write the attribute lmdbs
        for i, group in enumerate(attri_idx_groups):
            label_datum = lu.generate_array_datum(group, is_float=False)
            attri_lmdbs[i][1].put(key, label_datum.SerializeToString())

        if count % 10000 == 0:
            # Write to lmdb every 10000 times
            img_txn.commit()
            bbox_txn.commit()
            img_txn = img_env.begin(write=True, buffers=True)
            bbox_txn = bbox_env.begin(write=True, buffers=True)
            for attri_lmdb in attri_lmdbs:
                attri_lmdb[1].commit()
                attri_lmdb[1] = attri_lmdb[0].begin(write=True, buffers=True)

    if cfg.debug_gen_flag is False:
        # Close the lmdb
        img_txn.commit()
        bbox_txn.commit()
        img_env.close()
        bbox_env.close()

        for attri_lmdb in attri_lmdbs:
            attri_lmdb[1].commit()
            attri_lmdb[0].close()

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

    csv_file.write('image_file, label, bbox_x1, bbox_y1, bbox_x2, bbox_y2, labels,\n')
    fail_count = 0

    for count, entry in enumerate(list):
        # Print info every 100 iterations
        if count % 100 == 0:
            print 'Currently processing on %d' % count

        # Extract infomations
        basic_info = entry if aug_flag is False else entry[0]
        image_name = basic_info[0]
        attri_labels = basic_info[1]
        bbox = np.asarray((basic_info[2])).flatten()
        offset_factor = np.asarray(basic_info[3], dtype=float).flatten()
        ext_factor = np.asarray(basic_info[4], dtype=float).flatten()
        minimal_width_ratio = float(basic_info[5])
        comp_idx = int(basic_info[6])

        flip_flag = False if aug_flag is True else bool(entry[1])
        trans_magnitude = (0, 0) if aug_flag is False else entry[2]
        zooming_magnitude = 0 if aug_flag is False else entry[3]

        # Create attribute label matrix
        attri_idx_groups = []
        for group in range(0, len(cfg.attributes_index)):
            group_len = cfg.attributes_index[group]['count']
            labels = np.zeros(group_len+1, dtype=int)
            labels[0] = 1      # Default index: Background
            attri_idx_groups.append(labels)

        # Read image
        img, img_size = None, None
        if attri_labels is not None:
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
                print 'Current:' + image_name
                cv.imshow('debug_preview', img)
                bar_plt.show_label_bar(attri_idx_groups, bar_fig)
                cv.imshow('bar_preview', cv.imread('test.png'))
                cv.waitKey(0)
                # pass
            continue


        # Check the bbox
        if ext_bbox[3] < ext_bbox[1] or ext_bbox[2] < ext_bbox[0]:
            fail_count += 1
            continue

        # Crop the image
        crop_img = img[int(ext_bbox[1]): int(ext_bbox[3]),
                       int(ext_bbox[0]): int(ext_bbox[2])]
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

        # Write to csv file
        csv_file.write('%s, %f, %f, %f, %f, %d, ' % (image_name,
                                                          ext_bbox[0], ext_bbox[1], ext_bbox[2], ext_bbox[3],
                                                          flip_flag))
        # Write multiple attributes
        if attri_labels is not None:
            for attribute in attri_labels:
                group_idx = attribute['group_idx']
                attri_idx = attribute['attri_idx']

                csv_file.write('%d-%d;' % (group_idx, attri_idx))
        else:
                csv_file.write('Bg_label')

        csv_file.write(',\n')

    csv_file.close()

    if cfg.debug_gen_flag is False:
        print 'Push items into csv file has finished, \nTotal: %d\n' % len(list)


def generate_entries(sql_querys, attribute_config, data_model):
    # Variables
    entry_lists = []

    # Iterate the SQL queries to get entries
    for component, infos in cfg.sql_queries.iteritems():

        # local variables
        total_counts = 0
        ignore_counts = 0

        # Extract information from configuration
        label_groups = infos['groups']
        label_trans_idx = infos['trans']

        # Run the query to get datas
        sql_query = infos['query']
        bbox_points = infos['points']
        component_id = infos['comp_id']
        raw_data = data_model.query(sql_query)
        for data in raw_data:

            # Extract image name
            image_name = data[0]

            # Extract bounding boxes
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

            # Extract labels
            attri_labels = []
            trans_label_name = None
            idx_offset = 2*bbox_points + 1
            for i in range(0, len(label_groups)):

                # Extract label name
                label_name = data[idx_offset + i]
                group_idx = label_groups[i]

                # Check the replace table if need to be replaced by other label
                if cfg.attri_replace_table.has_key(label_name):
                    label_name = cfg.attri_replace_table[label_name]

                if group_idx == label_trans_idx:
                    trans_label_name = label_name

                # Search the attribute informations
                # Check if contains key
                if cfg.attributes_index[group_idx].has_key(label_name):
                    attri_idx = cfg.attributes_index[group_idx][label_name][0][1]
                else:
                    attri_idx = 0
                    continue

                attri_labels.append({'group_idx':group_idx, 'attri_idx': attri_idx, 'label_name':label_name})

            # Search the attribute informations
            if not cfg.attributes_index[label_trans_idx].has_key(trans_label_name):
                ignore_counts += 1
                continue

            trans_infos = cfg.attributes_index[label_trans_idx][trans_label_name][1]

            attri_offset = (float(trans_infos[0]), float(trans_infos[1]))
            attri_ext_factor = (float(trans_infos[2]), float(trans_infos[3]))
            attri_ratio = float(trans_infos[4])

            # Put together to entry
            entry = (image_name, attri_labels,  # Basic information
                     (start_pt, end_pt),        # Bounding box
                     attri_offset,              # Relative Offset
                     attri_ext_factor,          # Extend factor
                     attri_ratio,               # Minimal ratio of height/width
                     component_id
                     )

            # Add to entry list
            entry_lists.append(entry)
            total_counts += 1

        print 'Component:' + component + ' total:' + str(total_counts) + ' ignored:' + str(ignore_counts) + ' images.'

    # Shuffle the 'entry_list'
    shuffle(entry_lists)

    # If we need clamp the dataset:
    # set cfg.maximal_total_images equals to -1 will not clamp the dataset
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
        entry = (neg_img, None,             # Basic information
                (start_pt, end_pt),         # Bounding box
                (0.0, 0.0),                 # Relative Offset
                (1.0, 1.0),                 # Extend factor
                1.0,                         # Minimal ratio of height/width
                0
                )

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


def merge_entries(entries_list):

    # Add to a dictionary to reduce repeat items
    entries_dict = {}
    for entry in entries_list:
        img_name = entry[0]

        if entries_dict.has_key(img_name):
            entries_dict[img_name].append(img_name)
        else:
            entries_dict[img_name] = []

    # Analysis and Create new entry that covers whole image
    i = 0
    while i < len(entries_dict):
        img_name = entries_dict.keys()[i]
        entry_set = entries_dict[img_name]

        # Iterate the entry to get the maximal bounding box
        start_pt = [sys.float_info.max, sys.float_info.max]
        end_pt = [sys.float_info.min, sys.float_info.min]

        # If not negative image

        for entry in entry_set:

            bbox = np.asarray(entry[2]).flatten()

            if entry[1] is not None:
                # Extend the crop region with random zooming
                ext_bbox = au.extend_bbox_xy(bbox, img_size,
                                             ext_factor[0] * (1 + zooming_magnitude), ext_factor[1] * (1 + zooming_magnitude),
                                             xy_ratio=minimal_width_ratio)

                ext_bbox_size = (ext_bbox[2] - ext_bbox[0], ext_bbox[3] - ext_bbox[1])

                # Apply random translate
                random_offset = (trans_magnitude[0] * ext_bbox_size[0], trans_magnitude[1] * ext_bbox_size[1])
                ext_bbox = au.translate_box(ext_bbox, img_size, random_offset)
            else:



    return None

if __name__ == '__main__':

    """[TEST] preview window initialization
    """
    if cfg.debug_gen_flag is True:

        # Create preview Window
        cv.namedWindow('debug_preview')
        cv.namedWindow('bar_preview')

        # Check if we need generate debug preview files, we need to create
        # directory
        if cfg.enable_debug_gen_file is True and os.path.exists(cfg.debug_gen_path) is False:
            os.mkdir(cfg.debug_gen_path)

    """Main pipeline for generate lmdbs
    """
    # Add PyCaffe to sys.path
    sys.path.append(cfg.pycaffe_path)

    # Create data provider
    data_model = GarmentDataModel(cfg.mysql_hostname,
                                  cfg.mysql_username,
                                  cfg.mysql_userpasswd,
                                  cfg.mysql_dbname)

    # Parsing the data into training, validation, test list
    ori_training_list, ori_validate_list, ori_test_list = generate_entries(cfg.sql_queries,
                                                                           cfg.attributes_index,
                                                                           data_model)

    print 'Debug'

    # # Generate test lists
    # print '> Writing training list to file: ' + cfg.lmdb_output_path + 'train_list.txt'
    # generate_csvfile(ori_training_list, cfg.lmdb_output_path + 'train_list.txt')
    #
    # print '> Writing validation list to file: ' + cfg.lmdb_output_path + 'val_list.txt'
    # generate_csvfile(ori_validate_list, cfg.lmdb_output_path + 'val_list.txt')
    #
    # print '> Writing test list to file: ' + cfg.lmdb_output_path + 'test_list.txt'
    # generate_csvfile(ori_test_list, cfg.lmdb_output_path + 'test_list.txt')
    #
    # # Augment the training dataset
    # aug_training_list = augment_training_data(ori_training_list)
    # print '> Augmented ' + str(len(aug_training_list)) + ' training images'
    #
    # aug_validate_list = augment_training_data(ori_validate_list)
    # print '> Augmented ' + str(len(aug_validate_list)) + ' validate images'
    #
    # # Generate lmdbs
    # print 'Start to create training lmdbs.'
    # # generate_lmdbs(ori_training_list, cfg.train_img_dbpath, cfg.train_bbox_dbpath, lmdb_prefix='train', aug_flag=False)
    # generate_lmdbs(aug_training_list, cfg.train_img_dbpath, cfg.train_bbox_dbpath, lmdb_prefix='train', aug_flag=True)
    #
    # print 'Start to create validation lmdbs.'
    # generate_lmdbs(aug_validate_list, cfg.valid_img_dbpath, cfg.valid_bbox_dbpath, lmdb_prefix='valid', aug_flag=True)

    # Destory window if needed
    if cfg.debug_gen_flag is True:
        cv.destroyAllWindows()
