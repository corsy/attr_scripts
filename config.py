#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Configuration attributes
"""
import math


"""
    Caffe configuration
"""
pycaffe_path = '/opt/caffe/python'

"""
    MySQL connection configurations
"""
mysql_hostname = 'localhost'
mysql_username = 'root'
mysql_userpasswd = 'nesfsm'
mysql_dbname = 'clothesimagedatabase'

"""
    Dataset configurations
"""
# Set the maximal image amount in dataset, if -1 means no limitation
maximal_total_images = -1

# Set the directories
image_directory = '/home/luwei/Project/Datasets/Garment/database20141024/'
trousers_image_directory = '/home/luwei/Project/Datasets/Garment/database-trousers/'

# Negative image directories
neg_image_directory = '/home/luwei/Project/Datasets/INRIA/'

# Set max negative image will be used
max_neg_img_count = 200

# Resize dimension
output_img_size = (227, 227)

"""
    Attributes file configurations
    attribute file is the table that contains information in database

"""
attribute_file_path = 'attri_config.csv'

"""
    Output directories and path
"""
# Configure output lmdbs path
lmdb_output_path = ''

# Path for outputting training image lmdb
train_img_dbpath = lmdb_output_path + 'image_train.lmdb'

# Path for outputting training bbox lmdb
train_bbox_dbpath = lmdb_output_path + 'bbox_train.lmdb'

# Configure how many images will be used as training set
train_imgs_ratio = 0.6

# Path for outputting validation image lmdb
valid_img_dbpath = lmdb_output_path + 'image_valid.lmdb'

# Path for outputting validation bbox lmdb
valid_bbox_dbpath = lmdb_output_path + 'bbox_valid.lmdb'

# Configure how many images will be used as validate set
valid_imgs_ratio = 0.2

# Configure test list path
test_list_path = lmdb_output_path + 'test_img_list.txt'

# Configure how many test image you want to put into test list
test_imgs_ratio = 0.2

"""
    Dataset and augmentation configuration
"""

# Debug flag of dataset generating
debug_gen_flag = False

# Enable or disable generate the debug data to file, if 'debug_gen_flag' is True
enable_debug_gen_file = False

# Debug generate dataset output path
debug_gen_path = lmdb_output_path + 'debug_preview'

# Zooming the bounding box to 1.3x
base_bbox_zooming_factor = 1.3

# Set the factor of augment
# e.g. if set 'augment_factor' = 20, their will be
# 20 more images per training item
augment_size = 20

# Set what percentage of data need to be flipped, if set 'flip_percent' = 1.0
# all item will be flipped, default = 0.5
flip_percent = 0.5

# Set distribution method
# the distribution can be 'Gaussian' or 'Uniform'
distribution_method = 'Gaussian'

# Set what magnitude of data need to be randomly translation, relative to
# the bounding box size
translate_factor = 0.20

# Set what magnitude of data need to be randomly zooming, relative to
# the bounding box size
zooming_factor = 0.20

# Set what percentage of data need to be randomly rotate
# TODO(Luwei): Implemnet rotate augmentation later
rotate_factor = 1.0

# Set the maximum of rotation, the rotate range is
# (-rotate_max_angle, rotate_max_angle)
rotate_max_angle = math.pi / 2

# Set what percentage of data need to have color jitter
# TODO(Luwei): Implemnet color jitter later
color_jitter_percent = 1.0

# Set what magnitude of color jitter will be applied to original image bitmap.
color_jitter_factor = 1.0
