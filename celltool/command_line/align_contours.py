# Copyright 2007 Zachary Pincus
# This file is part of CellTool.
#
# CellTool is free software; you can redistribute it and/or modify
# it under the terms of version 2 of the GNU General Public License as
# published by the Free Software Foundation.

"""Align a set of contours to one another or to a specific reference.

This tool takes a set of contours -- which must all have the same number of
points -- and aligns them to one another, or to a separate, fixed reference.
This reference can either be another contour, or a PCA shape model, in which
case the mean contour shape from the model is used for alignment.

Contour alignment is a non-trivial problem because two parameters must be
found: first, the physical rotation, translation, and (optionally) reflection
which maximally brings the contour points into alignment, and second, the
ordering of those contour points which permits the best such alignment. (If
the contour points are not ordered in an analogous manner from contour to
contour -- such that point number 'n' from one contour corresponds to the
physical location of point number 'm' from another contour, then the best
physical alignment between the sets of points is not particularly
informative.) Thus we jointly optimize the point-correspondences between
contours and their physical alignments.

The general alginment procedure, given a fixed reference is thus:
    - for several drastically different candidate point orderings:
            find the best physical alignment of points to the reference
    - keep the candidate ordering that gives the best overall alignment
    - directly optimize this candidate ordering
That is, it is a rough global optimization followed by a local optimization.

For simple shapes, a small number of initial candidate orderings will be
sufficient -- 4 should do. For more complex shapes which might be difficult to
align, more candidate orderings -- 8 or 16 -- may be necessary. If the
alignment quality is poor, consider increasing the number.

If there is no fixed reference, then the contours are initially aligned to
their long axes, and then the mean contour is calculated as a reference for
alignment. After aligning each contour to the mean, the mean is then
re-computed and the procedure is repeated. (This is the 'expectation
maximization' algorithm.) Iteration terminates when no contours change (or
after ten iterations, whichever comes first).
"""

import optparse
from celltool import simple_interface
from celltool.utility import path
from celltool.contour import contour_class
from . import cli_tools

usage = "usage: %prog [options] contour_1 ... contour_n"

parser = optparse.OptionParser(usage=usage, description=__doc__.strip(),
    formatter=cli_tools.CelltoolFormatter())
parser.set_defaults(
    show_progress=True,
    allow_reflection=False,
    alignment_steps=8,
    max_iterations=10,
    destination='.'
)
parser.add_option('-q', '--quiet', action='store_false', dest='show_progress',
    help='suppress progress bars and other status updates')
parser.add_option('-a', '--allow-reflection', action='store_true', dest='allow_reflection',
    help='allow contours to be reflected over some axis if it improves the alignment')
parser.add_option('-s', '--alignment-steps', type='int', metavar='STEPS',
    help='number of candidate point orderings to try for each shape during global alignment [default: %default]')
parser.add_option('-m', '--max-iterations', type='int', metavar='ITERS',
    help='maximum number of iterations for mutual contour alignment [default: %default]')
parser.add_option('-r', '--reference', metavar='CONTOUR',
    help='reference contour file that other contours will be aligned to (if not specified, contours will be mutually aligned)')
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
    if options.reference is not None:
        reference = contour_class.from_file(options.reference)
        contours = simple_interface.align_contours_to(contours, reference, align_steps=options.alignment_steps,
            allow_reflection=options.allow_reflection, show_progress=options.show_progress)
    else:
        contours = simple_interface.align_contours(contours, options.alignment_steps, options.allow_reflection,
            max_iters=options.max_iterations, show_progress = options.show_progress)
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