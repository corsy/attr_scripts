#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Configuration attributes
"""

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



