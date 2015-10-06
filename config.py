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
maximal_total_images = 200

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

attri_index_file_path = 'attri_index.csv'

"""
    Output directories and path
"""
# lmdb groups
attri_lmdb_groups = 4

# attribute lmdb prefix
attri_lmdb_prefix = 'attri_'

# Configure output lmdbs path
lmdb_output_path = './lmdb/'

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
debug_gen_flag = True

# Enable or disable generate the debug data to file, if 'debug_gen_flag' is True
enable_debug_gen_file = False

# Debug generate dataset output path
debug_gen_path = lmdb_output_path + 'debug_preview'

# Zooming the bounding box to 1.3x
base_bbox_zooming_factor = 1.3

# Set the factor of augment
# e.g. if set 'augment_factor' = 20, their will be
# 20 more images per training item
augment_size = 4

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


"""
    SQL Queries
"""
sql_queries = {

    'Skirt': { 'query': 'SELECT clothingimagetag.ID_Image, p1_x, p1_y, p2_x, p2_y, SkirtLength, SkirtShape, SkirtPleat \
                         FROM clothingimagetag, skirtlabel \
                         WHERE clothingimagetag.ID_image=skirtlabel.ID_image \
                         AND SkirtLength!="None"',
               'points': 2,
               'groups': (0, 1, 2),         # Contains information in 'attributes_index'
               'trans' : 0                  # Refer to Group 0 for bbox transformation information
             },

    'Collar': { 'query': 'SELECT clothingimagetag.ID_Image, p1_x, p1_y, p2_x, p2_y, CollarType  \
                         FROM clothingimagetag, collarlabel \
                         WHERE clothingimagetag.ID_image=collarlabel.ID_image \
                         ',
               'points': 2,
               'groups': (4),               # Contains information in 'attributes_index'
               'trans' : 4                  # Refer to Group 0 for bbox transformation information
             },

    'Placket': { 'query': 'SELECT clothingimagetag.ID_Image, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y,  \
                         p5_x, p5_y, p6_x, p6_y,Placket1   \
                         FROM clothingimagetag, upperlabel \
                         WHERE clothingimagetag.ID_image=upperlabel.ID_image \
                         ',
               'points': 6,
               'groups': (5),               # Contains information in 'attributes_index'
               'trans' : 5                  # Refer to Group 0 for bbox transformation information
             },

    'Sleeve': { 'query': 'SELECT clothingimagetag.ID_Image, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y,  \
                         p5_x, p5_y, p6_x, p6_y, SleeveLength  \
                         FROM clothingimagetag, sleevelabel \
                         WHERE clothingimagetag.ID_image=sleevelabel.ID_image \
                         ',
               'points': 6,
               'groups': (6),               # Contains information in 'attributes_index'
               'trans' : 6                  # Refer to Group 0 for bbox transformation information
             },

    'Button': { 'query': 'SELECT clothingimagetag.ID_Image, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, \
                         p5_x, p5_y, p6_x, p6_y ButtonType \
                         FROM clothingimagetag, upperlabel \
                         WHERE clothingimagetag.ID_image=upperlabel.ID_image \
                         ',
               'points': 6,
               'groups': (7),               # Contains information in 'attributes_index'
               'trans' : 7                  # Refer to Group 0 for bbox transformation information
             },

}

"""
    Attribute Indices
"""
attributes_index = [
    # name     # (group_idx, label_idx),  # bbox transformation
                                          # (x_offset; y_offset; x_ext_factor; y_ext_factor, x_y_ratio)

    # Group 1: Skirt length
    {
    'Chang'         : [(0, 1),               (0,    0, 1.1, 1.1, 0.8)],
    'Duan'          : [(0, 2),               (0, 0.05, 1.2, 1.6, 0.8)],
    'Zhong'         : [(0, 3),               (0, 0.05, 1.2, 1.4, 0.8)],
    },

    # Group 2: Shape of Skirt
    {
    'Denglongzhuang': [(1, 1),               None],                      # 'None' means no transformation
    'Labazhuang'    : [(1, 2),               None],
    'Zhitongzhuang' : [(1, 3),               None],
    },

    # Group 3: Pleat of Skirt
    {
    'None'          : [(2, 1),               None],
    'You'           : [(2, 2),               None],
    },

    # Group 4: Collar type
    {
    'Fanling'       : [(4, 1),               (0, 0, 1.8, 1.8, 0.8)],
    'Liling'        : [(4, 2),               (0, 0, 1.8, 1.8, 0.8)],
    'None'          : [(4, 3),               (0, 0, 1.8, 1.8, 0.8)],
    },

    # Group 5: Placket type
    {
    'Duijin'        : [(5, 1),               (0, 0, 0.7, 0.5, 0.7)],
    'None'          : [(5, 2),               (0, 0, 0.7, 0.5, 0.7)],
    },

    # Group 6: Sleeve Length
    {
    'Changxiu'      : [(6, 1),           (0, -0.05, 1.0, 1.0, 0.6)],
    'Duanxiu'       : [(6, 2),            (0, 0.11, 2.0, 3.0, 0.6)],
    },

    # Group 7: Button Type
    {
    'Danpaikou'     : [(7, 1),            (0, -0.1, 0.7, 0.6, 0.8)],
    'Lalian'        : [(7, 2),            (0, -0.1, 0.7, 0.6, 0.8)],
    'None'          : [(7, 3),            (0, -0.1, 0.7, 0.6, 0.8)],
    }
]

attri_replace_table = {
    # ori   # replacee
    'Dajin':'Duijin',
}