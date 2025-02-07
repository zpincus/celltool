# Copyright 2007 Zachary Pincus
# This file is part of CellTool.
# 
# CellTool is free software; you can redistribute it and/or modify
# it under the terms of version 2 of the GNU General Public License as
# published by the Free Software Foundation.

"""Plot a the distribution of one or more data sets in one or two dimensions.

This tool generates an SVG plot of the 1D or 2D distribution of one or more
data sets. Each data set is specified as a comma- or tab-separated data file.
If one data column is specified, then the 1-dimensional marginal distribution
of each data set, along that column, is plotted. If two columns are specified,
then the point distribution along both axes (as a scatterplot) is plotted for
each data set.

In the two-column case, the actual shape of each contour that corresponds to
each row in the given data set is plotted at the (x,y) position specified by
the values of the two data columns selected.

Data columns can be specified by number (the first column is numbered 1), or
if the data file has a header row, by name. The names of the contour files are
expected to be in the first column by default, but this can be configured as 
well.

It is possible to separate a single data file into multiple data sets. If one
or more '-r' or '--range' options are specified on the command-line
immediately after the name od a data file, that file is broken into multiple
data sets, each containing the rows specified in the range (inclusive). Rows
are indexed starting from one. For example, to split 'data.csv' into two data
sets, use something like: ... data.csv --range 1 100 "top cells" --range 101
200 "bottom cells" ... note that ranges must be named so that they can be
disambiguated. """

import optparse
from celltool.plot import plot_tools
from celltool.utility import path
from celltool.contour import contour_class
from celltool.utility import datafile
from celltool.utility import warn_tools
from celltool.utility import terminal_tools
from . import cli_tools
import numpy

def handle_range(option, opt_str, value, parser):
    data_file = parser.largs.pop()
    if type(data_file) != list:
        data_file = [data_file, []]
    low, high, name = value
    try:
        low = int(low)
        high = int(high)
    except:
        raise optparse.OptionValueError('The first two arguments to "%s" must be integers.'%opt_str)
    if low < 1 or high <= low:
        raise optparse.OptionValueError('The first two arguments to "%s" must define a (low, high) range, where low >=1 and high > low.'%opt_str)
    data_file[-1].append((low, high, name))
    parser.largs.append(data_file)

usage = "usage: %prog [options] data_file_1 [range] ... data_file_n [range] contour_1 ... contour_n"

parser = optparse.OptionParser(usage=usage, description=__doc__.strip(),
    formatter=cli_tools.CelltoolFormatter())
parser.set_defaults(
    show_progress=True,
    name_column = 1,
    x_column = 2,
    axes_at_origin=True,
    output_file='distribution-plot.svg',
    contour_axes=False
)
parser.add_option('-q', '--quiet', action='store_false', dest='show_progress',
    help='suppress progress bars and other status updates')
parser.add_option('-t', '--title', 
    help='title for the plot [default none]')
parser.add_option('-n', '--name-column', metavar='COL',
    help='look for contour names in this column (either a number or the column name) [default %default]')
parser.add_option('-x', '--x-column', metavar='COL',
    help='use data from this column index or name as x-values [default %default]')
parser.add_option('--x-title', metavar='TITLE',
    help='title to add to the x-axis [if not specified, try to read from data file]')
parser.add_option('--x-min', type='float', metavar='MIN',
    help='minimum value of the x-axis [if not specified, fit to data]')
parser.add_option('--x-max', type='float', metavar='MAX',
    help='maximum value of the x-axis [if not specified, fit to data]')
parser.add_option('-y', '--y-column', metavar='COL',
    help='use data from this column index or name as y-values [if not specified, plot is 1D]')
parser.add_option('--y-title', metavar='TITLE',
    help='title to add to the y-axis [if not specified, try to read from data file]')
parser.add_option('--y-min', type='float', metavar='MIN',
    help='minimum value of the y-axis (only used for 2D plots -- in 1D case, y range is automatic) [if not specified, fit to data]')
parser.add_option('--y-max', type='float', metavar='MAX',
    help='maximum value of the y-axis (only used for 2D plots -- in 1D case, y range is automatic) [if not specified, fit to data]')
parser.add_option('-l', '--lower-left-axes', action='store_false', dest='axes_at_origin',
    help='place the axes at the lower-left of the plot [by default axes intersect at the origin]')
parser.add_option('-r', '--range', action='callback', callback=handle_range, 
    nargs=3, type='str', metavar='LOW HIGH NAME',
    help='define a named range of rows in the data file immediately previous.')
parser.add_option('-s', '--scale', type='float',
    help='for 2D plots: pixels per contour-unit scaling factor (plot will be 640x480) [if not specified, a reasonable scale is chosen]')
parser.add_option('--contour-axes', action='store_true',
    help='Draw central axes on contours if present')
parser.add_option('-o', '--output-file', metavar='FILE',
    help='SVG file to write [default: %default]')

def main(name, arguments):
    parser.prog = name
    options, args = parser.parse_args(arguments)
    data_files = []
    new_args = []
    for arg in args:
        if type(arg) == list:
            data_files.append(arg)
        else:
            new_args.append(arg)
    args = cli_tools.glob_args(new_args)
    if len(args) + len(data_files) == 0:
        raise ValueError('Some data files (and optionally contour files, for 2D plots) must be specified!')
    
    # if the x, y, and name columns are convertible to integers, do so and 
    # then convert them from 1-indexed to 0-indexed
    try:
        options.name_column = int(options.name_column)
        options.name_column -= 1
    except:
        pass
    try:
        options.x_column = int(options.x_column)
        options.x_column -= 1
    except:
        pass
    try:
        options.y_column = int(options.y_column)
        options.y_column -= 1
    except:
        pass
    contours = {}
    if options.show_progress:
        args = terminal_tools.progress_list(args, "Reading input data and contours")
    for arg in args:
        contour = None
        try:
            contour = contour_class.from_file(arg)
        except:
            data_files.append(arg)
        if contour is not None:
            contours[contour.simple_name()] = contour
    if len(data_files) == 0:
        raise ValueError('No data files were specified!')
    headers, data_ranges, data_names, data_files, row_ranges = get_data(data_files)
    if not options.x_title:
            if isinstance(options.x_column, int):
                try: options.x_title = headers[0][options.x_column]
                except: pass
            else:
                options.x_title = options.x_column
    if options.y_column is not None:
        # make scatterplot
        if not options.y_title:
            if isinstance(options.y_column, int):
                try: options.y_title = headers[0][options.y_column]
                except: pass
            else:
                options.y_title = options.y_column
        contour_groups = get_contour_groups(data_ranges, contours, data_files, row_ranges, options.name_column, options.x_column, options.y_column)
        if numpy.all([len(cg)==0 for cg in contour_groups]):
            raise RuntimeError('No contours found for data rows specified (perhaps the names mismatch or there were no data rows?).')
        plot_tools.contour_scatterplot(contour_groups, options.output_file, options.scale,
            (options.x_title, options.y_title), options.title, names=data_names, 
            axes_at_origin=options.axes_at_origin, fix_xrange=(options.x_min, options.x_max),
            fix_yrange=(options.y_min, options.y_max), show_contour_axes=options.contour_axes, 
            show_progress=options.show_progress)
    else:
        data_groups = [[row[options.x_column] for row in data_range] for data_range in data_ranges]
        plot_tools.distribution_plot(data_groups, options.output_file, options.x_title,
            options.title, names=data_names, axes_at_origin=options.axes_at_origin, 
            fix_xrange=(options.x_min, options.x_max))

def get_data(data_files):
    data_ranges = []
    data_names = []
    new_data_files = []
    row_ranges = []
    headers = []
    for df in data_files:
        if type(df) == list:
            df, ranges = df
        else:
            ranges = None
        # read the data file, and don't try to convert column 0 -- the contour names --
        # to anything other than a string.
        data = datafile.DataFile(df, skip_empty = False, type_dict = {0:str})
        header, rows = data.get_header_and_data()
        headers.append(header)
        if ranges is None:
            data_ranges.append(rows)
            data_names.append(path.path(df).namebase)
            new_data_files.append(df)
            if header is None: start = 0
            else: start = 1
            row_ranges.append(list(range(1+start, len(data.data)+start)))
        else:
            for low, high, name in ranges:
                try:
                    # recall that low, high give an inclusive, one-indexed range...
                    data_ranges.append(data.data[low-1:high])
                except:
                    raise ValueError('Cannot get row range (%d, %d) from data file "%s" -- not enough rows?')
                data_names.append(name)
                new_data_files.append(df)
                row_ranges.append(list(range(low, high+1)))
    return headers, data_ranges, data_names, new_data_files, row_ranges

def get_contour_groups(data_ranges, contours, data_files, row_ranges, name_column, x_column, y_column):
    contour_groups = []
    for data_range, data_file, row_range in zip(data_ranges, data_files, row_ranges):
        try:
            data_range[0][name_column], data_range[0][x_column], data_range[0][y_column]
        except:
            raise ValueError('Cannot find specified columns in data file "%s".'%data_file)
        contour_and_points = []
        for i, row in zip(row_range, data_range):
            if numpy.all([r == None for r in row]):
                continue
            contour_name, x, y = row[name_column], row[x_column], row[y_column]
            if not contour_name:
                warn_tools.warn('Could not find contour name in otherwise non-empty row (file "%s", row %d).'%(data_file, i))
                continue
            if type(x) not in (int, float):
                warn_tools.warn('Cannot read numeric value from column %s in file "%s", row %d.'%(x_column, data_file, i))
                continue
            if type(y) not in (int, float):
                warn_tools.warn('Cannot read numeric value from column %s in file "%s", row %d.'%(y_column, data_file, i))
                continue    
            try:
                contour = contours[contour_name]
            except:
                raise ValueError('Cannot find contour file with name "%s" (referenced in file "%s", row %d).'%(contour_name, data_file, i))
            contour_and_points.append((contour, (x, y)))
        contour_groups.append(contour_and_points)
    return contour_groups

if __name__ == '__main__':
    import sys
    import os
    main(os.path.basename(sys.argv[0]), sys.argv[1:])


