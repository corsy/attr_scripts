#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This script will be responsible for handling
    parsing attribute configuration file
"""
import codecs

def get_attribute_index(index_file_path):
    """
    Get the attribute index
    :param index_file_path: path of index config file
    :return: dictionary of attribute_index
    """
    idx_file = open(index_file_path, 'r')

    line_count = 0
    field_names = None
    attribute_indices = {}

    # Read line-by-line to parsing attribute index files
    while 1:

        # Read into buffer to speed up
        lines = idx_file.readlines(100000)

        if not lines:
            break
        for line in lines:

            # Parsing first line, get the filed name
            if line_count == 0:
                field_names = [x.strip() for x in line.split(';')[1:-1]]
                line_count += 1
                continue

            # Parsing items to entry[table row]
            attribute_entry = {}
            field_values = [x.strip() for x in line.split(';')[:-1]]
            for i, field_name in enumerate(field_names):
                attribute_entry[field_name] = field_values[i+1]

            # Push to attribute indices
            attribute_indices[field_values[0]] = attribute_entry

            line_count += 1

    # Close configuration file
    idx_file.close()

    return attribute_indices


def parsing_attribute_configfile(config_file_path):

    """
    Parsing attributes configuration file

    CSV file format as follows:

        component,          # Component name
        attribute,          # Attribute name
        label_db_schema,    # Database schema that store the attribute label
        label_db_fieldname,
        label_db_fieldvalue,
        bbox_db_schema,     # Attribute corresponding bounding box schema
        bbox_db_pts,
        image_query,        # SQL query string used for searching image
        bbox_query          # SQL query string used for searching bounding box infomation

    INPUT:
       PARAMS            | TYPE                   | DESCRIPTION
    1. config_file_path  | String                 | Attributes configuration file path

    OUTPUT:
        PARAMS           | TYPE                   | DESCRIPTION
    1. attribute_list    | List                   | Attributes information table

    """

    # Open configuration file
    config_file = codecs.open(config_file_path, 'r', encoding='utf8')

    # Some variables
    line_count = 0
    field_names = None
    attribute_list = []

    # Read line-by-line to parsing attribute files
    while 1:

        # Read into buffer to speed up
        lines = config_file.readlines(100000)

        if not lines:
            break
        for line in lines:

            # Parsing first line, get the filed name
            if line_count == 0:
                field_names = line.split(';')
                line_count += 1
                continue

            # Parsing items to entry[table row]
            attribute_entry = {}
            field_values = line.split(';')
            for i, field_name in enumerate(field_names):
                attribute_entry[field_name] = field_values[i]

            # Push to attribute list
            attribute_list.append(attribute_entry)

            line_count += 1

    # Close configuration file
    config_file.close()

    return attribute_list
