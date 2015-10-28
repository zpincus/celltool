# Copyright 2007 Zachary Pincus
# This file is part of CellTool.
# 
# CellTool is free software; you can redistribute it and/or modify
# it under the terms of version 2 of the GNU General Public License as
# published by the Free Software Foundation.

"""Create a PCA shape model from a set of pre-aligned contours.

This tool applies the principal components analysis to a set of contours
(which should have been resampled to the same number of points and aligned to
one another previously), producing a PCA contour data file and optionally a 
list of the positions of each input contour along each of the principal modes.

Two comma-delimited files of contour positions will be written (unless disabled):
the absolute positions of each contour along each axis (in arbitrary units) and
the normalized positions of each contour along each axis, in units of standard
deviations along that axis. Thus, the normalized positions provide an equivalent
scaling for each shape mode. Note however, that this can be misleading, as a 
mode which explains much of the total variance will be scaled the same as a mode
that explains little. Thus, when performing multivariate comparisons of shapes
along multiple modes, it is best to use the non-normalized absolute positions, 
as these take into account the relative importance of each mode. However for
comparisons involving only one shape mode and other, unrelated, data, the
normalized modes are perfectly useful.

Without a limit, there will be as many PCA shape modes as there are contours,
or twice the number of points in the contours, whichever is less. However,
most of these modes will be relatively unimportant; thus typically only the 
first few modes are retained. That number is chosen as the smallest number of 
modes which collectively explain more than some threshold fraction of the total
variance in the data set. This threshold can be manually set, though the default
of 0.95 works well.
"""

import optparse
from celltool import simple_interface
from celltool.contour import contour_class
from celltool.utility import datafile
from . import cli_tools

usage = "usage: %prog [options] contour_1 ... contour_n"

parser = optparse.OptionParser(usage=usage, description=__doc__.strip(),
    formatter=cli_tools.CelltoolFormatter())
parser.set_defaults(
    show_progress=True,
    variance_explained=0.95,
    output_prefix='shape-model',
    write_data=True,
)
parser.add_option('-q', '--quiet', action='store_false', dest='show_progress',
    help='suppress progress bars and other status updates')
parser.add_option('-v', '--variance-explained', type='float',
    help='minimum fraction of total variance explained by the recorded shape modes [default: %default]')
parser.add_option('-n', '--no-data', action='store_false', dest='write_data',
    help='do not write out the positions and normalized positions of the contours')
parser.add_option('-o', '--output-prefix', 
    help='directory and file name prefix for shape model and data files written [default: %default]')

def main(name, arguments):
    parser.prog = name
    options, args = parser.parse_args(arguments)
    args = cli_tools.glob_args(args)
    if len(args) == 0:
        raise ValueError('Some contour files must be specified!')
    contours = simple_interface.load_contours(args, show_progress = options.show_progress)
    shape_model, header, rows, norm_header, norm_rows = simple_interface.make_shape_model(contours, options.variance_explained)
    shape_model.to_file(options.output_prefix + '.contour')
    if options.write_data:
        datafile.write_data_file([header]+rows, options.output_prefix + '-positions.csv')
        datafile.write_data_file([norm_header]+norm_rows, options.output_prefix + '-normalized-positions.csv')

if __name__ == '__main__':
    import sys
    import os
    main(os.path.basename(sys.argv[0]), sys.argv[1:])