# Copyright 2007 Zachary Pincus
# This file is part of CellTool.
# 
# CellTool is free software; you can redistribute it and/or modify
# it under the terms of version 2 of the GNU General Public License as
# published by the Free Software Foundation.

"""Geometrically modify a set of contours.

This tool can be used to alter contours by rotating or rescaling them, or by
changing the ordering of their points.
If the contours have landmark points, these points can be re-weighted by this
tool. (See the documentation for add_landmarks for more details.)
"""

import optparse
from celltool import simple_interface
from celltool.utility import path
from . import cli_tools

usage = "usage: %prog [options] contour_1 ... contour_n"

parser = optparse.OptionParser(usage=usage, description=__doc__.strip(),
    formatter=cli_tools.CelltoolFormatter())
parser.set_defaults(
    show_progress=True,
    destination='.'
)
parser.add_option('-q', '--quiet', action='store_false', dest='show_progress',
    help='suppress progress bars and other status updates')
parser.add_option('-r', '--rotate', type='float', metavar='ANGLE',
    help='rotate each contour counter-clockwise by ANGLE (in degrees)')
parser.add_option('-s', '--scale', action='store', type='float', metavar='FACTOR',
    help='rescale the contours by multiplicative FACTOR')
parser.add_option('-u', '--units', action='store',
    help='name of the (new) units for the contours')
parser.add_option('-f', '--first-point', type='int', metavar='POINT',
    help='make point number POINT the new first point in the contour points')
parser.add_option('-w', '--weight', action='append', dest='weights', 
    type='float', metavar='WEIGHT',
    help='set the weight shared by all contours to WEIGHT (if specified once), or set the weight of the nth landmark to WEIGHT (if specified multiply)')
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
    if options.first_point is not None:
        options.first_point -= 1
    if options.scale is not None or options.rotate is not None or options.units is not None or options.first_point is not None:
        in_radians = False
        if options.units is not None and options.units.lower() in ('um', 'micron', 'microns'):
            options.units = '\N{MICRO SIGN}m'
        contours = simple_interface.transform_contours(contours, options.scale, options.rotate, in_radians, 
            options.units, options.first_point, options.show_progress, title = 'Modifying Contours')
    if options.weights is not None:
        contours = simple_interface.reweight_landmarks(contours, options.weights, options.show_progress)
        
    
    destination = path.path(options.destination)
    destination.makedirs_p()
    # note that with path objects, the '/' operator means 'join path components.'
    names = [destination / filename.name for filename in filenames]
    simple_interface.save_contours(contours, names, options.show_progress)

if __name__ == '__main__':
    import sys
    import os
    main(os.path.basename(sys.argv[0]), sys.argv[1:])