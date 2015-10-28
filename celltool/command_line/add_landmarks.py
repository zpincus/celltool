# Copyright 2007 Zachary Pincus
# This file is part of CellTool.
# 
# CellTool is free software; you can redistribute it and/or modify
# it under the terms of version 2 of the GNU General Public License as
# published by the Free Software Foundation.

"""Add landmark points to contours from a given set of images.

In some cases, the contour outline alone is insufficient to properly align
certain shapes. The alignment procedure can be aided by adding a set of
explicit landmark points to the contours. 

Landmark points for each contour are read from images. This is so that it is
possible to generate the landmark points manually by paining over images, or
automatically with thresholding or some other image processing step. Different
landmarks are differentiated by different pixel intensity values, so that it 
is possible to add multiple landmarks from one image.

Given a pixel intensity range, an image, and a contour, the position of the 
landmarks are found as follows:
  - Find the region in the image that corresponds to the interior of the
    contour. If the images provided are in the same orientation as those from
    which the contours were originally derived, this works automatically. 
    However, if the images were extracted with the extract_images tool (or are
    derived from images so extracted), it is necessary to specify this fact by
    setting the '-a' or '--aligned-images' flag.
  - Within that region, find all of the pixels that fall into the intensity
    range. The landmark is defined as the geometric centroid of these pixels.

Once one or more landmarks have been read from an image, they are associated
with the contour file that is then written out. If all of the contour files
passed to the align_contours tool have landmarks (and have the same number of
landmarks!), then these will be used in the alignment procedure.

Finally, it is possible to weight the landmarks, to control the relative
influence of each landmark versus the others, and versus the contour points.
Weights are specified with the '-w' option. If there is no weight specified, 
then the landmarks (in aggregate) will share half of the total weight; the
other half will be shared by the contour points. If there is one '-w' value
specified, then the landmarks will share that fraction of the total weight, 
and the points the remaining amount. (Thus the '-w' value must be <= 1.) If
multiple '-w' values are provided, then there must be exactly as many weights
as landmarks; moreover, the sum of the weights must be <= 1.

Landmarks can be specified as either a single pixel-intensity values or as two
values, which are treated as an inclusive [low, high] intensity interval. To 
add a landmark of the former type, use the '-l' or '--landmark' option; for
the latter use the '-i' or '--landmark-interval' option.

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
  image_type='original',
  destination='.'
)
parser.add_option('-q', '--quiet', action='store_false', dest='show_progress',
  help='suppress progress bars and other status updates')
parser.add_option('-l', '--landmark', action='append', dest='intensity_ranges', 
  type='int', metavar='INTENSITY',
  help='find pixels with values equal to INTENSITY and treat their centroid as a landmark')
parser.add_option('-i', '--landmark-interval', action='append', dest='intensity_ranges', 
  type='int', nargs=2, metavar='LOW HIGH',
  help='find pixels with values in [LOW, HIGH] and treat their centroid as a landmark')
parser.add_option('-w', '--weight', action='append', dest='weights', 
  type='float', metavar='WEIGHT',
  help='set the weight shared by all contours to WEIGHT (if specified once), or set the weight of the nth landmark to WEIGHT (if specified multiply)')
parser.add_option('-r', '--order-match', action='store_false', dest='match_by_name',
  help='match contours to images by their order on the command line [default: match by name]')
parser.add_option('-a', '--aligned-images', action='store_const', const='aligned', dest='image_type',
  help='set if images have been aligned with extract_images [default: images are assumed to be oriented as the original contours were]')
parser.add_option('-d', '--destination', metavar='DIRECTORY',
  help='directory in which to write the output contours [default: %default]')

def main(name, arguments):
  parser.prog = name
  options, args = parser.parse_args(arguments)
  args = cli_tools.glob_args(args)
  if len(args) == 0:
    raise ValueError('Some contour and image files must be specified!')
  intensity_ranges = []
  for r in options.intensity_ranges:
    try:
      low, high = r
    except:
      low = high = r
    intensity_ranges.append([low, high])
  if options.weights is None or len(options.weights) == 0:
    options.weights = [0.5]
  elif len(options.weights) != 1 and len(options.weights) != len(intensity_ranges):
    raise optparse.OptionValueError("Either one or %d weights are required; %d sepcified."%(len(intensity_ranges), len(options.weights)))
  elif sum(options.weights) > 1:
    raise optparse.OptionValueError("The sum of the weights must be <= 1.")
  
  matches = match_files.match_contours_and_images(args, options.match_by_name, options.show_progress)
  contours, image_names, unmatched_contours, unmatched_image_names = matches
  filenames = [path.path(contour._filename) for contour in contours]
  contours = simple_interface.add_image_landmarks_to_contours(contours, image_names,
    intensity_ranges, options.weights, options.image_type, options.show_progress)
  destination = path.path(options.destination)
  if not destination.exists():
    destination.makedirs()
  # note that with path objects, the '/' operator means 'join path components.'
  names = [destination / filename.name for filename in filenames]
  simple_interface.save_contours(contours, names, options.show_progress)


if __name__ == '__main__':
  import sys
  import os
  main(os.path.basename(sys.argv[0]), sys.argv[1:])