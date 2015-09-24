#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Utility of data augmentation

    This script will handling augmentation of training datasets
    including randomly tranlation and randomly rotate, random zooming.

"""

def extend_bbox(bbox, img_size, extend_factor):
    """
    Extend the region of bounding box by the extend scale

    @param(bbox): bounding box, including start and end point
    @param(img_size): dimension of image, used when extend out of boundary.
    @param(extend_factor): extend scale, relative to the size of original
    bounding box.

    @return(ext_bbox): extended bounding box in (x1, y1, x2, y2)

    """

    size = (bbox[2] - bbox[0], bbox[3] - bbox[1])       # size(width,height)
    center = (bbox[0] + size[0]/2, bbox[1] + size[1]/2) # center (x,y)

    # Compute extended bounding box
    ext_start_x = center[0] - extend_factor * size[0]/2
    ext_start_x = 0 if ext_start_x < 0

    ext_end_x = center[0] + extend_factor * size[0]/2
    ext_end_x = img_size[0] if ext_end_x > img_size[0]

    ext_start_y = center[1] - extend_factor * size[1]/2
    ext_start_y = 0 if ext_start_y < 0

    ext_end_y = center[1] + extend_factor * size[1]/2
    ext_end_y = img_size[1] if ext_end_y > img_size[1]

    ext_bbox = (ext_start_x, ext_start_y, ext_end_x, ext_end_y)

    return ext_bbox
