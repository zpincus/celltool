# Copyright 2007 Zachary Pincus
# This file is part of CellTool.
#
# CellTool is free software; you can redistribute it and/or modify
# it under the terms of version 2 of the GNU General Public License as
# published by the Free Software Foundation.
import re
import freeimage

from celltool.contour import contour_class
from celltool.utility import terminal_tools
from celltool.utility import warn_tools
from celltool.utility import path

def match_contours_and_images(filenames, match_by_name, show_progress = True, warn_unmatched_contour = True, warn_unmatched_image = True):
    contours, image_names = _get_contours_and_images(filenames, show_progress)
    if not match_by_name:
        if len(contours) != len(image_names):
            raise ValueError('Cannot match %d contours to %d images by order -- there must be an equal number.' %(len(contours), len(image_names)))
        return contours, image_names, [], []
    image_shortnames = {}
    for image_name in image_names:
        name = image_name.namebase
        if name in image_shortnames:
            raise ValueError('Two images named %s were provided. Image names must be unique.'%name)
        image_shortnames[name] = image_name
    new_contours = []
    new_images = []
    unmatched_contours = []
    for contour in contours:
        name = contour.simple_name()
        no_trailing_numbers = _remove_trailing_numbers(name)
        if name in image_shortnames:
            # If the name matches exactly, remove the image from the available set
            new_contours.append(contour)
            image_name = image_shortnames.pop(name)
            new_images.append(image_name)
        elif no_trailing_numbers in image_shortnames:
            # If the name matches after removing numbers, keep the image available for others
            new_contours.append(contour)
            image_name = image_shortnames[no_trailing_numbers]
            new_images.append(image_name)
        else:
            unmatched_contours.append(contour)
    unmatched_images = [image for image in image_names if image not in new_images]
    if warn_unmatched_contour:
        for contour in unmatched_contours:
            warn_tools.warn('Contour "%s" not matched to any image.'%contour._filename)
    if warn_unmatched_image:
        for image_name in unmatched_images:
            warn_tools.warn('Image "%s" not matched to any contour.'%image_name)
    return new_contours, new_images, unmatched_contours, unmatched_images

_number_finder =    re.compile(r'^(.*)(?:-(\d+))$') # matches strings than end in -[digits]
def _remove_trailing_numbers(name):
    m = _number_finder.match(name)
    if m is None:
        return name
    else:
        return m.group(1)

def _get_contours_and_images(filenames, show_progress = True):
    contours = []
    image_names = []
    if show_progress:
        filenames = terminal_tools.progress_list(filenames, 'Loading Contours and Images')
    for filename in filenames:
        filename = path.path(filename)
        if not filename.exists():
            raise ValueError('File "%s" does not exist.'%filename)
        try:
            freeimage.read_metadata(filename)
            image_names.append(filename)
        except IOError, e:
            # print e
            # print Image.ID
            try:
                contours.append(contour_class.from_file(filename))
            except IOError, e:
                # print e
                raise ValueError('Could not open file "%s" as an image or a contour.'%filename)
    return contours, image_names
