#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    LMDB utinity

    This script contains useful routines to handle lmdb database

"""

import os
import lmdb
import cv2 as cv
import shutil
import numpy as np
import caffe

def del_and_create(database_file_path):

    """
        Delete the exist lmdb database and create new lmdb database.
    """

    if os.path.exists(database_file_path):
        shutil.rmtree(database_file_path)
    os.mkdir(database_file_path)

def generate_img_datum(img, label):

    """
        Put the data of a array into Caffe's datum, this datum will be used in lmdb.

        INPUT:
           PARAMS            | TYPE                   | DESCRIPTION
        1. img               | Numpy Array            | The array of image bitmap.
        2. label             | Integer                | The corresponding label for image.

        OUTPUT:
           PARAMS            | TYPE                   | DESCRIPTION
        1. datum             | Caffe dataum           | Dataum that restore the image and label data

    """

    img = img.swapaxes(0, 2).swapaxes(1, 2)
    datum = caffe.io.array_to_datum(img, 0)
    datum.label = label

    return datum

def generate_array_datum(array):

    """
        Put the data of a array into Caffe's datum, this datum will be used in lmdb.

        INPUT:
           PARAMS            | TYPE                   | DESCRIPTION
        1. array             | Array                  | The array need to put into a datum unit.

        OUTPUT:
           PARAMS            | TYPE                   | DESCRIPTION
        1. datum             | Caffe dataum           | Dataum that restore the data of array

    """

    array = np.asarray(array).flatten()

    datum = caffe.io.caffe_pb2.Datum()
    datum.channels = len(array)
    datum.height = 1
    datum.width = 1
    datum.float_data.extend(array.tolist())

    return datum
