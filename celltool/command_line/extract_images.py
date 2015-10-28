# Copyright 2007 Zachary Pincus
# This file is part of CellTool.
# 
# CellTool is free software; you can redistribute it and/or modify
# it under the terms of version 2 of the GNU General Public License as
# published by the Free Software Foundation.

"""Extract image regions corresponding to previously-extracted contours.

This tool trims and rotates the region of an image corresponding to the region
from which a contour was extracted, and saves that to a new image. For
example, if a contour was extracted from a binary mask image, itself made from
a image of a fluorescent protein distribution, this tool could be used to
extract that the protein distribution in the area of the contour and re-orient
it to the contour's orientation. Thus, the tool can be used to re-orient image
regions to a canonical orientation. Optionally, the region outside of the
contour can be masked to black in the resulting image.

All extracted image regions will be made the same size -- the size of the
largest contour (plus a small amount of padding) -- so they are directly
comparable.

For this tool to work with multiple images and contours, it must be able to
match images to contour files. This can be accomplished in two ways: 

(1) By name: An image is matched to a contour file if their names match
exactly, except for the file extensions and an optional trailing
hyphen-and-numbers for the contour file. (The hyphen-and-numbers allows for
multiple unique contours to derive from, and match to, a single image.) This
is the default.

(2) By order: The first contour file matches the first image file, and so
forth. There must be exactly as many contour files as image files. To match by
order, use the '-r' or '--order-match flag'.

In both cases, the resulting image file is named the same as the contour.
"""

from celltool.utility import optparse
from celltool import simple_interface
from celltool.utility import path
import cli_tools
import match_files

usage = "usage: %prog [options] image_or_contour_1 ... image_or_contour_n"

parser = optparse.OptionParser(usage=usage, description=__doc__.strip(),
  formatter=cli_tools.CelltoolFormatter())
parser.set_defaults(
  show_progress=True,
  match_by_name=True,
  mask_background=False,
  destination='.',
  pad_factor=1.1
)
parser.add_option('-q', '--quiet', action='store_false', dest='show_progress',
  help='suppress progress bars and other status updates')
parser.add_option('-r', '--order-match', action='store_false', dest='match_by_name',
  help='match contours to images by their order on the command line [default: match by name]')
parser.add_option('-m', '--mask-background', action='store_true',
  help='set the image intensity outside of the contour to zero')
parser.add_option('-p', '--pad-factor', metavar='VALUE', type='float',
  help='the fractional amount of extra padding around the largest contour to include in the images [default: %default]')
parser.add_option('-f', '--file-type', choices=('tif', 'tiff', 'png'), metavar='TYPE',
  help='image type to write out (tiff or png; if not specified use type of input image)')
parser.add_option('-d', '--destination', metavar='DIRECTORY',
  help='directory in which to write the output contours [default: %default]')

def main(name, arguments):
  parser.prog = name
  options, args = parser.parse_args(arguments)
  args = cli_tools.glob_args(args)
  if len(args) == 0:
    raise ValueError('Some contour and image files must be specified!')
  matches = match_files.match_contours_and_images(args, options.match_by_name, options.show_progress)
  contours, image_names, unmatched_contours, unmatched_image_names = matches
  destination = path.path(options.destination)
  if not destination.exists():
    destination.makedirs()
  if options.file_type is not None:
    new_names = [destination / contour.simple_name() + '.' + options.file_type for contour in contours]
  else:
    new_names = [destination / contour.simple_name() + image.ext for image, contour in zip(image_names, contours)]
  simple_interface.reorient_images(contours, image_names, new_names, pad_factor=options.pad_factor, 
    mask=options.mask_background, show_progress=options.show_progress)

if __name__ == '__main__':
  import sys
  import os
  main(os.path.basename(sys.argv[0]), sys.argv[1:])