# Copyright 2007 Zachary Pincus
# This file is part of CellTool.
# 
# CellTool is free software; you can redistribute it and/or modify
# it under the terms of version 2 of the GNU General Public License as
# published by the Free Software Foundation.

"""Plot a set of contours, optionally showing the ordering of their points.

This tool generates an SVG plot of the outlines of a set of contours.
Optionally, the contours can be colored by their point ordering, to allow a
visual inspection of the point correspondences between the contours.
"""

import optparse
from celltool.plot import plot_tools
from celltool import simple_interface
import cli_tools

usage = "usage: %prog [options] contour_1 ... contour_n"

parser = optparse.OptionParser(usage=usage, description=__doc__.strip(), 
    formatter=cli_tools.CelltoolFormatter())
parser.set_defaults(
    show_progress=True,
    label_points=False,
    color_by='contours',
    grid=False,
    output_file='contour-plot.svg'
)
parser.add_option('-q', '--quiet', action='store_false', dest='show_progress',
    help='suppress progress bars and other status updates')
parser.add_option('-t', '--title', 
    help='title for the plot [default none]')
parser.add_option('-b', '--begin', type='int', metavar='POINT',
    help='contour position at which to begin plots [if not specified, contour start]')
parser.add_option('-e', '--end', type='int', metavar='POINT',
    help='contour position at which to end plots [if not specified, contour end]')
parser.add_option('-c', '--color-by', type='choice', choices=('contours','points', 'none'),
    help='how to color the plotted contours: by "contours" , by "points", or "none" [default: %default]')
parser.add_option('-l', '--label-points', action='store_true',
    help=' show explicit labeled point correspondences')
parser.add_option('-s', '--scale', type=float,
    help='pixels per contour-unit scaling factor [if not specified, a reasonable default is chosen]')
parser.add_option('--grid', action='store_true',
    help='display all contours in a 2D grid')
parser.add_option('-o', '--output-file', metavar='FILE',
    help='SVG file to write [default: %default]')

def main(name, arguments):
    parser.prog = name
    options, args = parser.parse_args(arguments)
    args = cli_tools.glob_args(args)
    if len(args) == 0:
        raise ValueError('Some contour files must be specified!')
    contours = simple_interface.load_contours(args, show_progress = options.show_progress)
    if options.grid:
        contours = simple_interface.grid_contours(contours)
# Transform (begin, end) from a one-indexed, inclusive range (normal for humans)
# to a zero-indexed inclusive range.
    if options.begin is not None:
        options.begin -= 1
    if options.end is not None:
        options.end -= 1
    if options.color_by == 'none':
        gradient_factory = None
    else:
        gradient_factory = plot_tools.default_gradient
    if options.color_by == 'points' or options.begin or options.end or options.label_points:
        # do point ordering plot
        colorbar = options.color_by == 'points'
        color_by_point = options.color_by == 'points'
        # use a color bar if we're doing a gradient        
        plot_tools.point_order_plot(contours, options.output_file, plot_title=options.title, 
            label_points=options.label_points, colorbar=colorbar, begin=options.begin, 
            end=options.end, gradient_factory=gradient_factory, color_by_point=color_by_point,
            scale=options.scale, show_progress=options.show_progress)
    else:
        plot_tools.contour_plot(contours, options.output_file, plot_title=options.title, 
            gradient_factory=gradient_factory, scale=options.scale, show_progress=options.show_progress)

if __name__ == '__main__':
    import sys
    import os
    main(os.path.basename(sys.argv[0]), sys.argv[1:])


