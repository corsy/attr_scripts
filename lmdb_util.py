#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    LMDB utinity

    This script contains useful routines to handle lmdb database

"""

import os
import lmdb
import shutil

def del_and_create(database_file_path):
    if os.path.exists(database_file_path):
        shutil.rmtree(database_file_path)
    os.mkdir(database_file_path)


