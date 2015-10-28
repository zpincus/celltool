# Copyright 2007 Zachary Pincus
# This file is part of CellTool.
#
# CellTool is free software; you can redistribute it and/or modify
# it under the terms of version 2 of the GNU General Public License as
# published by the Free Software Foundation.

"""Tools to read and write numpy arrays from and to image files.
"""

import freeimage
import numpy
import warn_tools


def read_grayscale_array_from_image_file(filename, warn = True):
    """Read an image from disk into a 2-D grayscale array, converting from color if necessary.

    If 'warn' is True, issue a warning when arrays are converted from color to grayscale.
    """
    image_array = freeimage.read(filename)
    if len(image_array.shape) == 3:
        image_array = make_grayscale_array(image_array)
        if warn:
            warn_tools.warn('Image %s converted from RGB to grayscale: intensity values have been scaled and combined.'%filename)
    return image_array

write_array_as_image_file = freeimage.write

def make_grayscale_array(array):
    """Giiven an array of shape (x,y,3) where the last dimension indexes the
    (r,g,b) pixel value, return a (x,y) grayscale array, where intensity is
    calculated with the ITU-R BT 709 luma transform:
            intensity = 0.2126r + 0.7152g + 0.0722b
    """
    dtype = array.dtype
    new_array = numpy.round((array * [0.2126, 0.7152, 0.0722]).sum(axis = 2))
    return new_array.astype(dtype)
