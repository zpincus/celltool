# Copyright 2007 Zachary Pincus
# This file is part of CellTool.
# 
# CellTool is free software; you can redistribute it and/or modify
# it under the terms of version 2 of the GNU General Public License as
# published by the Free Software Foundation.

"""Plot the shape modes of a given PCA shape model.

This tool generates an SVG plot of the 'shape modes' of a given PCA shape
model. For each principal shape mode, the outlines of contours at each of
several positions along that mode are plotted in superimposition to give an
intuitive picture of the type of shape variation captured by that mode.

By default the positions are (-2sd, -1sd, the mean, 1sd, and 2sd), where 'sd'
indicates 'standard deviations away from the mean along the given mode'.

By default, each mode captured by the shape model will be plotted. Note that
shape modes are indexed from one (e.g. a 3-mode contour has modes 1, 2, and
3).
"""

from celltool.utility import optparse
from celltool.plot import plot_tools
from celltool.contour import contour_class
import cli_tools

usage = "usage: %prog [options] shape_model_file"

parser = optparse.OptionParser(usage=usage, description=__doc__.strip(),
  formatter=cli_tools.CelltoolFormatter())
parser.set_defaults(
  show_progress=True
)
parser.add_option('-q', '--quiet', action='store_false', dest='show_progress',
  help='suppress progress bars and other status updates')
parser.add_option('-m', '--mode', action='append', type='int', dest='modes',
  help='add a shape mode to the plot; to add multiple modes use this option multiple times [if none specified, all modes are used]')
parser.add_option('-p', '--position', action='append', type='float', dest='positions',
  help='add a position to the plot; to add multiple positions use this option multiple times [if none specified, positions -2,-1,0,1, and 2 are used]')
parser.add_option('-s', '--scale', type=float,
  help='pixels per contour-unit scaling factor [if not specified, a reasonable default is chosen]')
parser.add_option('-o', '--output-file', metavar='FILE',
  help='SVG file to write [defaults to the name of the shape model file specified]')

def main(name, arguments):
  parser.prog = name
  options, args = parser.parse_args(arguments)
  args = cli_tools.glob_args(args)
  if len(args) != 1:
    raise ValueError('Only a single PCA contour can be plotted at a time -- please specify only one model file on the command line.')
  pca_contour = contour_class.from_file(args[0], contour_class.PCAContour)
  if not options.output_file:
    output_file = pca_contour.simple_name()+'.svg'
  else:
    output_file = options.output_file
  plot_tools.pca_modes_plot(pca_contour, output_file, modes=options.modes, 
    positions=options.positions, scale=options.scale)
      
if __name__ == '__main__':
  import sys
  import os
  main(os.path.basename(sys.argv[0]), sys.argv[1:])


