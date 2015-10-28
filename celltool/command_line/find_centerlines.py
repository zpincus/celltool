# Copyright 2007 Zachary Pincus
# This file is part of CellTool.
# 
# CellTool is free software; you can redistribute it and/or modify
# it under the terms of version 2 of the GNU General Public License as
# published by the Free Software Foundation.

"""Find the center-lines of a set of of contours.

The center-line of a contour is defined to be a line from one position on the
contour to another position, such that the line is as smooth and evenly-
spaced as possible while at the same time being centered between the contour's
edges. ("Centered" is defined as a point being mid-way between the
intersections of the contour with a line normal to the axis at that point.)

So defined, centerlines are not unique; the best we can do is find the best
such line, given a starting and ending point for the central axis. Note that
the numerical optimiztion procedure to find the "best" center-line can also
move the endpoints, so what is needed is just a good guess as to their
locations.

This tool can guess these endpoint locations in several different ways:
    (1) The locations can be selected as the two points that are most physically
            distant. ("distance" method)
    (2) The locations can be selected as the top-most and bottom-most points on
            the contour. ("vertical" method)
    (3) The locations can be selected as the leftmost and rightmost points on
            the contour. ("horizontal" method; this is default)
    (4) The locations can be selected as the positions along the contour where
            the contour bends inward most sharply (subject to the constraint that
            the positions selected are separated by at least 1/3 the distance of
            the contour). ("curvature" method)

The best choice is strongly dependent on the specific shapes under
consideration. In general, the default "horizontal" method is a good choice
for contours that have been previously aligned along their long axes (as is
default for the align_contours command). Alternately, the starting and ending
points can be specified directly.
"""

from celltool.utility import optparse
from celltool import simple_interface
from celltool.utility import path
import cli_tools

usage = "usage: %prog [options] contour_1 ... contour_n"

parser = optparse.OptionParser(usage=usage, description=__doc__.strip(),
    formatter=cli_tools.CelltoolFormatter())
parser.set_defaults(
    show_progress=True,
    destination='.',
    axis_points=25,
    endpoint_method='horizontal'
)
parser.add_option('-q', '--quiet', action='store_false', dest='show_progress',
    help='suppress progress bars and other status updates')
parser.add_option('-p', '--axis-points', type='int', metavar='POINTS',
    help='number of points to calculate along the central axis [default: %default]')
parser.add_option('-m', '--endpoint-method', type='choice', metavar='METHOD',
    choices=['horizontal', 'vertical', 'distance', 'curvature'],
    help='method to use to guess the location of the center-line endpoints [default "%default"] (not used if endpoints are specified directly with the --endpoints option)')
parser.add_option('-e', '--endpoints', type='float', nargs=2, metavar='POSITION',
    help='starting and ending vertex numbers (two numbers must be specified)')
parser.add_option('-d', '--destination', metavar='DIRECTORY',
    help='directory in which to write the output contours [default: %default]')

def main(name, arguments):
    parser.prog = name
    options, args = parser.parse_args(arguments)
    args = cli_tools.glob_args(args)
    if len(args) == 0:
        raise ValueError('Some contour files must be specified!')
    filenames = [path.path(arg) for arg in args]
    contours = simple_interface.load_contours(filenames, show_progress = options.show_progress)
    if options.endpoints is None:
        endpoints = options.endpoint_method
    else:
        endpoints = options.endpoints
    contours = simple_interface.find_centerlines(contours, options.axis_points, endpoints, options.show_progress)
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