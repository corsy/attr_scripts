#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Configuration attributes
"""
import math

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
# Set the image amount in dataset
dataset_total_images = 5580

# Set the directories
image_directory = '/home/luwei/Project/Datasets/Garment/database20141024/'
trousers_image_directory = '/home/luwei/Project/Datasets/Garment/database-trousers/'

"""
    Attributes file configurations
    attribute file is the table that contains information in database

"""
attribute_file_path = '/home/luwei/Project/Datasets/Garment/attri_config.csv'

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
train_size = 2000

# Path for outputting validation image lmdb
valid_img_dbpath = lmdb_output_path + 'image_valid.lmdb'

# Path for outputting validation bbox lmdb
valid_bbox_dbpath = lmdb_output_path + 'bbox_valid.lmdb'

# Configure how many images will be used as validate set
validation_size = 2000

# Configure test list path
test_list_path = ''

# Configure how many test image you want to put into test list
test_size = dataset_total_images - (train_size + validation_size)

"""
    Dataset and augmentation configuration
"""
# Zooming the bounding box to 1.3x
base_bbox_zooming_factor = 1.3

# Set the factor of augment
# e.g. if set 'augment_factor' = 20, their will be
# 20 more images per training item
augment_factor = 20

# Set what percentage of data need to be flipped, if set 'flip_percent' = 1.0
# all item will be flipped, default = 0.5
flip_percent = 0.5

# Set distribution method
# the distribution can be 'Gaussian' or 'Uniform'
distribution_method = 'Gaussian'

# Set what percentage of data need to be randomly translation
translate_factor =

# Set what percentage of data need to be randomly rotate
rotate_factor =

# Set the maximum of rotation, the rotate range is
# (-rotate_max_angle, rotate_max_angle)
rotate_max_angle = math.pi / 2

#
