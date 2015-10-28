# Copyright 2007 Zachary Pincus
# This file is part of CellTool.
# 
# CellTool is free software; you can redistribute it and/or modify
# it under the terms of version 2 of the GNU General Public License as
# published by the Free Software Foundation.

"""Extract contours from images, optionally rescaling and resampling them.

This tool extracts contours from images at a given threshold value. Optionally
a minimum and maximum area for acceptable contours can be specified.

Once extracted, contours can be rescaled so that they are defined in terms of
some spatial length scale (instead of pixels). If a scale is specified, it is
highly recommended to also specify the units of that scale. (Note that
'microns' or 'um' will be converted to the micron symbol.)

After scaling, the contours can be re-sampled to smooth them and ensure that
each contour has the same number of points, and that those points are evenly
spaced. Smoothing is controlled by a smoothing parameter, which sets a maximum
on the mean squared-distance between the points of the original contour and
the points of the smoothed contour. This distance will be in terms of the true
spatial units, if specified.

If the primary source of measurement error in the shapes is from pixel 
quantization, then a smoothing factor on the order of the squared distance
between pixels is appropriate. This would be 1 for un-scaled contours, or 
the square of the units-per-pixel value otherwise.

Contours will be named based on the images that they were extracted from. If
there were multiple contours for a given image, then a number will be appended
to the image name.
"""

from celltool.utility import optparse
from celltool import simple_interface
from celltool.utility import path
import cli_tools

usage = "usage: %prog [options] image_1 ... image_n"

parser = optparse.OptionParser(usage=usage, description=__doc__.strip(),
  formatter=cli_tools.CelltoolFormatter())
parser.set_defaults(
  show_progress=True,
  min_area=10,
  units='',
  resample=True,
  resample_points=100,
  smoothing_factor=0,
  destination='.'
)
parser.add_option('-q', '--quiet', action='store_false', dest='show_progress',
  help='suppress progress bars and other status updates')
parser.add_option('-v', '--contour-value', action='store', type='float', metavar='VALUE',
  help='intensity level at which to extract contours [default: use the mid-point intensity of each image]')
parser.add_option('--min-area', action='store', type='float', metavar='AREA',
  help='minimum area for extracted contours; those smaller will be rejected [default: %default]')
parser.add_option('--max-area', action='store', type='float', metavar='AREA',
  help='maximum area for extracted contours; those larger will be rejected')  
parser.add_option('-s', '--scale', action='store', type='float',
  help='size of one pixel in spatial units (if specified, contours will be scaled in terms of those units)')
parser.add_option('-u', '--units', action='store',
  help='name of the units in which contours are measured [default: "pixels" if no scale is specified, otherwise none]')
parser.add_option('-n', '--no-resample', action='store_false', dest='resample',
  help='do not resample/smooth contours')
parser.add_option('-p', '--resample-points', type='int', metavar='POINTS',
  help='number of points in each contour after resampling (if resampling is enabled) [default: %default]')
parser.add_option('-f', '--smoothing-factor', type='float', metavar='SMOOTHING',
  help='maximum mean-squared-distance between original and resampled points (if resampling is enabled) [default: %default]')
parser.add_option('-d', '--destination', metavar='DIRECTORY',
  help='directory in which to write the output contours [default: %default]')

def main(name, arguments):
  parser.prog = name
  options, args = parser.parse_args(arguments)
  args = cli_tools.glob_args(args)
  if len(args) == 0:
    raise ValueError('Some image files must be specified!')
  filenames = [path.path(arg) for arg in args]
  contours_groups = simple_interface.extract_contours(filenames, options.contour_value, 
    options.min_area, options.max_area, options.show_progress)
  contours = []
  names = []
  destination = path.path(options.destination)
  if not destination.exists():
    destination.makedirs()
  for contour_group, image_name in zip(contours_groups, filenames):
    num_contours = len(contour_group)
    if num_contours == 1:
      contours.append(contour_group[0])
      # note that with path objects, the '/' operator means 'join path components.'
      names.append(destination / image_name.namebase + '.contour')
      contour_group[0]._filename = image_name.namebase
    else:
      width = len(str(num_contours))
      for i, contour in enumerate(contour_group):
        contours.append(contour)
        names.append(destination / image_name.namebase + '-%.*d.contour'%(width, i+1))
        contour._filename = image_name.namebase + '-%.*d'%(width, i+1)
  if options.scale is not None:
    # if not rescaling, contours are already denominated in pixels, so do nothing.
    units = options.units
    if units.lower() in ('um', 'micron', 'microns'):
      units = u'\N{MICRO SIGN}m'
    contours = simple_interface.transform_contours(contours, scale_factor=options.scale, 
      units=units, show_progress=options.show_progress, title='Rescaling Contours')
  if options.resample:
    contours = simple_interface.resample_contours(contours, options.resample_points, options.smoothing_factor, options.show_progress)
  simple_interface.save_contours(contours, names, options.show_progress)

if __name__ == '__main__':
  import sys
  import os
  main(os.path.basename(sys.argv[0]), sys.argv[1:])