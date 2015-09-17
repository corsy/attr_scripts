#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This script will be responsible for handling
    parsing attribute configuration file
"""
import codecs



def parsing_attribute_configfile(config_file_path):

    """
    Parsing attributes configuration file
    INPUT: attributes configuration file path
    OUTPUT: attributes information table
    """

    # Open configuration file
    config_file = codecs.open(config_file_path, 'r', encoding='utf8')

    # Some variables
    line_count = 0
    field_names = None
    attribute_list = []

    # Read line-by-line to parsing attribute files
    while 1:
        lines = config_file.readlines(100000)
        if not lines:
            break
        for line in lines:

            # Parsing first line, get the filed name
            if line_count == 0:
                field_names = line.split(',')
                line_count += 1
                continue

            # Parsing items to entry[table row]
            attribute_entry = {}
            field_values = line.split(',')
            for i, field_name in enumerate(field_names):
                attribute_entry[field_name] = field_values[i]

            # Push to attribute list
            attribute_list.append(attribute_entry)

            line_count += 1

    # Close configuration file
    config_file.close()

    return attribute_list

"""
    Test parsing attributes
"""
import config as cfg
if __name__ == '__main__':
    attributes = parsing_attribute_configfile(cfg.attribute_file_path)

    print 'attributes:', len(attributes)
