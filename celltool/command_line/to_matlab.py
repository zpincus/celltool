# Copyright 2007 Zachary Pincus
# This file is part of CellTool.
#
# CellTool is free software; you can redistribute it and/or modify
# it under the terms of version 2 of the GNU General Public License as
# published by the Free Software Foundation.

"""Save contour data as Matlab-readable .mat files.

The files will be saved with the same name as the contour files in the
specified directory.

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
    destination = path.path(options.destination)
    destination.makedirs_p()
    # note that with path objects, the '/' operator means 'join path components.'
    names = [destination / filename.namebase + '.mat' for filename in filenames]
    simple_interface.save_contour_data_for_matlab(contours, names, options.show_progress)


if __name__ == '__main__':
    import sys
    import os
    main(os.path.basename(sys.argv[0]), sys.argv[1:])