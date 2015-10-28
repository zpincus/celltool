# Copyright 2007 Zachary Pincus
# This file is part of CellTool.
# 
# CellTool is free software; you can redistribute it and/or modify
# it under the terms of version 2 of the GNU General Public License as
# published by the Free Software Foundation.

"""Measure a set of contours in terms of shape and image intensity.

A suite of tools are available to measure any given contour's shape, global
and local geometry, and the relationship between shape and patterns of image
intensity. These measurements are then written out to a comma-delimited file.
As many measurements as desired may be specified on the command line. This
simply adds more data columns to the output file.

Some of these measurements are relatively complex and require sub-options of
their own to be fully-specified. These sub-options are specified on the
command line in the usual way, except that after the list of sub-options, a
single dash ('-') is required to terminate the list of sub-options. (Else how
would the program know which were sub-options and which were top-level
options.) Note that even if you do not wish to specify any sub-options, the
(empty) list still MUST be terminated with a '-'.

For example, to measure curvature from point 10 to point 30, indicate: ...
--curvature --begin=10 --end=30 - ... To accept the default begin and end
values (which cover the whole contour), use: ... --curvature - ... All of the
"usual" operations are valid, as sub-options are parsed and specified in the
same way. Note that --curvature='--help' will print the help text for the
curvature sub-option and exit, for example.

Available Measurements:
    (1) Contour area -- squared area within each contour (in the contour units).
    (2) Aspect ratio -- the width / height ratio of each contour.
    (3) Path length -- the distance along the contour in contour units. The
            beginning and ending points of the path can be specified via
            sub-options.
    (4) Normalized curvature -- The average of the absolute values of the
            pointwise curvature of the contour is computed over a specified range,
            and multiplied by the contour length over the same range. This is a
            basic measure of "roughness". Absolute values must be used because
            otherwise positive and negative curvatures would cancel out; the sum is
            multiplied by the arc length to make the measurement scale-invariant. (A
            large circle and a small circle have different curvatures, despite both
            being perfectly smooth; dividing by arc length ensures that any circle
            will have a "roughness" value of one.)
    (5) Shape modes -- given a PCA shape model, the positions of each contour 
            along a specified set of the PCA shape modes can be calculated. (Either
            the normalized positions, in units of standard deviations, or the raw
            positions can be reported.)    Note that shape modes are indexed from one 
            (e.g. a 3-mode contour has modes 1, 2, and 3).
    (6) Image swath -- a 'swath' of a specified depth can be traced out along
            a contour from a beginning point to an ending point. Image intensities
            falling within this swath can be averaged along the length or depth
            dimension, or both. Such measurements require image files to be matched
            with contour files -- see the 'extract_images' documentation for more 
            discussion on this.
            If the images provided are in the same orientation as those from which
            the contours were originally derived, this works automatically. However,
            if the images were extracted with the extract_images tool (or are
            derived from images so extracted), it is necessary to specify this fact
            by setting the '-a' or '--aligned-images' flag.
            Image swaths are always measured in the direction of increasing contour
            point numbers: therefore, if the end point specified is less than the
            begining point, the swath will be measured from the beginning point to
            the final point of the contour, then wrap around back to the ending
            point specified. (E.g. for a 100-point contour, the swath from 91 to 10
            covers twenty points: 91-100 and 1-10, not 91 down to 10.)
    (7) Integrated image intensity within a contour. See above for caveats about
            matching images with contours.
    (8) Centroid in x and y (in physical units, if provided).
    (9) The alignment angle of the contour. This is the angle (in degrees) by
            which the contour in its current alignment must be rotated to return it
            to its original alignment in the image from which it was extracted.
 (10) The size of the contour's bounding rectangle in x and y.
 (11) The normalized curvature along two discontiguous regions of the contour.

Various measurements are also availble for contours with central axes (created
with the find_centerlines tool):
    (1) Axis RMSD -- the root-mean-squared deviation of the points along the 
            central axis from the baseline defined by the endpoints of that axis.
            Note that the deviations are re-centered around zero before the RMSD is
            calculated, so that axes with deviations only on one side of the
            baseline can be easily compared with those that oscillate to both sides.
    (2) Relative Axis RMSD -- as above, but divided by the length of the 
            baseline.
    (3) Axis Length Ratio -- length of the central axis divided by the length of 
            the baseline.
    (4) Peak Axis Amplitude -- The farthest distance that the central axis gets
            from the baseline determined by its endpoints. The deviations are re-
            centered as above.
    (5) Axis Wavelength -- An estimate of the "wavelength" of the axis. Extrema
            of the axis are found (in terms of distance from the baseline), and the
            average inter-extremal distance is calculated, giving a half-wavelength.
            This value is multiplied by two and reported. (Note that axis endpoints
            are always counted as extrema.)
    (6) Axis Wavenumber -- An estimate of the number of full "cycles" the axis
            completes. This is one less than half of the number of axis extrema, as
            described above.
    (7) Axis Length -- The length of the central axis, optionally over a
            defined sub-region of the axis.
    (8) Axis Mean Diameter -- The average diameter of the contour along the
            central axis, optionally within a defined sub-region of that axis.
    (9) Axis Diameters -- The diameters of the contour along the central axis,
            optionally within a defined sub-region of that axis.
 (10) Normalized Axis Curvature -- The normalized curvatures of the axis (see
            above), optionally within a defined sub-region of that axis.
 (11) Axis Swath -- Sample an image along a region defined by the contour's
            axis. The swath goes horizontally along the length of the central axis
            (beginning and ending axis points can be specified), and vertically from
            the top to the bottom of the cell. The number of samples to take in this
            direction MUST be specified as the first argument to the sub-command.
            See the swath measurement explanation above for details about the other
            available options.
"""

import optparse
import sys
from celltool import simple_interface
from celltool.utility import warn_tools
from celltool.utility import datafile
from celltool.command_line import cli_tools


def append_const(option, opt_str, value, parser, const):
    dest = _get_dest(parser, option)
    dest.append(const)

def two_span_handler(option, opt_str, value, parser, measurement):
    dest = _get_dest(parser, option)
    b1, e1, b2, e2 = value
    dest.append((measurement, {'begin_1':b1-1, 'end_1':e1-1, 'begin_2':b2-1, 'end_2':e2-1}))    

begin_option=optparse.make_option('-b', '--begin', type='int', metavar='POINT',
    help='contour position at which to start measurements [if not specified, contour start]')
end_option=optparse.make_option('-e', '--end', type='int', metavar='POINT',
    help='contour position at which to end measurements [if not specified, contour end]')

begin_end_parser = optparse.OptionParser(option_list=[begin_option, end_option])
def begin_end_handler(option, opt_str, value, parser, measurement):
    begin_end_parser.prog = opt_str
    arguments = _find_arguments(parser)
    options, args = begin_end_parser.parse_args(arguments)
    if len(args) != 0:
        raise optparse.OptionValueError('Measurement "%s" does not take any positional arguments.'%opt_str)
    # Transform (begin, end) from a one-indexed, inclusive range (normal for humans)
    # to a zero-indexed inclusive range.
    if options.begin is not None:
        options.begin -= 1
    if options.end is not None:
        options.end -= 1
    arg_dict = {'begin':options.begin, 'end':options.end}
    dest = _get_dest(parser, option)
    dest.append((measurement, arg_dict))

def _get_dest(parser, option):
    try:
        dest = getattr(parser.values, option.dest)
    except:
        dest = None
    if dest is None:
        dest = []
        setattr(parser.values, option.dest, dest)
    return dest

def _find_arguments(parser):
    value = []
    while len(parser.rargs) != 0:
        arg = parser.rargs[0]
        if arg == '-':
            del parser.rargs[0]
            break
        else:
            value.append(arg)
            del parser.rargs[0]
    return value

shape_mode_parser = optparse.OptionParser(usage="usage: %prog [options] shape_model_file mode_1 ... mode_n")
shape_mode_parser.add_option('-r', '--raw-position', action='store_false', dest='normalized',
    default=True, help='report non-normalized shape positions [default is normalized]')

def shape_mode_handler(option, opt_str, value, parser):
    shape_mode_parser.prog = opt_str
    arguments = _find_arguments(parser)
    options, args = shape_mode_parser.parse_args(arguments)
    args = cli_tools.glob_args(args)
    if len(args) < 1:
        raise optparse.OptionValueError('Measurement "%s" requires a shape model to be specified.'%opt_str)
    shape_model = args[0]
    try:
        shape_modes = [int(arg) for arg in args[1:]]
    except:
        raise optparse.OptionValueError('Error parsing arguments for "%s": cannot convert shape modes "%s" to a list of integers.'%(opt_str, args[1:]))
    arg_dict = {'shape_model_file':shape_model, 'modes':shape_modes, 'normalized':options.normalized}
    dest = _get_dest(parser, option)
    dest.append((simple_interface.ShapeModeMeasurement, arg_dict))

swath_parser = optparse.OptionParser(usage="usage: %prog [options] image_1 ... image_n")
swath_parser.set_defaults(
    measurement_name='image swath',
    offset=0,
    depth=1,
    mode='grand_average',
    image_type='original',
    contour_match='name',
)
swath_parser.add_option(begin_option)
swath_parser.add_option(end_option)
swath_parser.add_option('-n', '--measurement-name',
    help='name for this measurement in the result file [default: %default]')
swath_parser.add_option('-o', '--offset', type='float',
    help='distance inward from contour edge at which to start the swath (outward if negative) [default: %default]')
swath_parser.add_option('-d', '--depth', type='float',
    help='distance inward from offset at which to stop the swath (outward if negative) [default: %default]')
swath_parser.add_option('-m', '--measurement-mode', choices=('depth_profile', 'length_profile', 'grand_average'), 
    metavar='MODE', dest='mode',
    help='how the image swath is averaged down (one of "depth_profile", "length_profile", or "grand_average") [default: %default]')
swath_parser.add_option('-s', '--samples', type='int', metavar='SAMPLES',
    help='number of samples taken along the inward swath [if not specified, equal to the depth]')
swath_parser.add_option('-a', '--aligned-images', action='store_const', const='aligned', dest='image_type',
    help='set if images have been aligned with extract_images [default: images are assumed to be oriented as the original contours were]')
swath_parser.add_option('-r', '--order-match', action='store_const', const='order', dest='contour_match',
    help='match contours to images by their order on the command line (see "extract_images" documentation) [default: match by name]')

def image_swath_handler(option, opt_str, value, parser):
    swath_parser.prog = opt_str
    arguments = _find_arguments(parser)
    options, args = swath_parser.parse_args(arguments)
    args = cli_tools.glob_args(args)
    if len(args) == 0:
        raise optparse.OptionValueError('Measurement "%s" requires at least some images to be specified.'%opt_str)
    if options.samples is None:
        samples = max(1, 2*int(round(options.depth))+1)
    else:
        samples = options.samples
    # Transform (begin, end) from a one-indexed, inclusive range (normal for humans)
    # to a zero-indexed inclusive range.
    if options.begin is not None:
        options.begin -= 1
    if options.end is not None:
        options.end -= 1
    
    arg_dict = {
        'measurement_name':options.measurement_name,
        'begin':options.begin,
        'end':options.end,
        'offset':options.offset,
        'depth':options.depth,
        'mode':options.mode,
        'samples':samples,
        'image_type':options.image_type,
        'contour_match':options.contour_match,
        'image_names':args
        }
    dest = _get_dest(parser, option)
    dest.append((simple_interface.SwathMeasurement, arg_dict))

axis_swath_parser = optparse.OptionParser(usage="usage: %prog [options] depth_samples image_1 ... image_n")
axis_swath_parser.set_defaults(
    measurement_name='axis swath',
    mode='grand_average',
    image_type='original',
    contour_match='name',
)
axis_swath_parser.add_option(begin_option)
axis_swath_parser.add_option(end_option)
axis_swath_parser.add_option('-n', '--measurement-name',
    help='name for this measurement in the result file [default: %default]')
axis_swath_parser.add_option('-m', '--measurement-mode', choices=('depth_profile', 'length_profile', 'grand_average'), 
    metavar='MODE', dest='mode',
    help='how the image swath is averaged down (one of "depth_profile", "length_profile", or "grand_average") [default: %default]')
axis_swath_parser.add_option('-a', '--aligned-images', action='store_const', const='aligned', dest='image_type',
    help='set if images have been aligned with extract_images [default: images are assumed to be oriented as the original contours were]')
axis_swath_parser.add_option('-r', '--order-match', action='store_const', const='order', dest='contour_match',
    help='match contours to images by their order on the command line (see "extract_images" documentation) [default: match by name]')

def axis_swath_handler(option, opt_str, value, parser):
    swath_parser.prog = opt_str
    arguments = _find_arguments(parser)
    options, args = axis_swath_parser.parse_args(arguments)
    args = cli_tools.glob_args(args)
    if len(args) < 2:
        raise optparse.OptionValueError('Measurement "%s" requires a sample count and at least one images to be specified.'%opt_str)
    try:
        samples = int(args[0])
    except:
        raise optparse.OptionValueError('Measurement "%s" requires that the number of samples to measure in the direction perpendicular to the axis be specified as the first argument.'%opt_str)
    # Transform (begin, end) from a one-indexed, inclusive range (normal for humans)
    # to a zero-indexed inclusive range.
    if options.begin is not None:
        options.begin -= 1
    if options.end is not None:
        options.end -= 1
    
    arg_dict = {
        'measurement_name':options.measurement_name,
        'mode':options.mode,
        'begin':options.begin,
        'end':options.end,
        'samples':samples,
        'image_type':options.image_type,
        'contour_match':options.contour_match,
        'image_names':args[1:]
        }
    dest = _get_dest(parser, option)
    dest.append((simple_interface.AxisSwathMeasurement, arg_dict))

integral_parser = optparse.OptionParser(usage="usage: %prog [options] image_1 ... image_n")
integral_parser.set_defaults(
    measurement_name='image integral',
    image_type='original',
    contour_match='name',
)
integral_parser.add_option('-n', '--measurement-name',
    help='name for this measurement in the result file [default: %default]')
integral_parser.add_option('-a', '--aligned-images', action='store_const', const='aligned', dest='image_type',
    help='set if images have been aligned with extract_images [default: images are assumed to be oriented as the original contours were]')
integral_parser.add_option('-r', '--order-match', action='store_const', const='order', dest='contour_match',
    help='match contours to images by their order on the command line (see "extract_images" documentation) [default: match by name]')

def image_integral_handler(option, opt_str, value, parser):
    integral_parser.prog = opt_str
    arguments = _find_arguments(parser)
    options, args = integral_parser.parse_args(arguments)
    args = cli_tools.glob_args(args)
    if len(args) == 0:
        raise optparse.OptionValueError('Measurement "%s" requires at least some images to be specified.'%opt_str)
    
    arg_dict = {
        'measurement_name':options.measurement_name,
        'image_type':options.image_type,
        'contour_match':options.contour_match,
        'image_names':args
        }
    dest = _get_dest(parser, option)
    dest.append((simple_interface.ImageIntegration, arg_dict))


sub_parsers = (
    (begin_end_parser, '-l / --path-length'),
    (begin_end_parser, '-c / --curvature'),
    (shape_mode_parser, '-s / --shape-modes'),
    (swath_parser, '-i / --image-swath'),
    (integral_parser, '-t / --integrate-image'),
    (begin_end_parser, '--axis-length'),
    (begin_end_parser, '--axis-mean-diameter'),
    (begin_end_parser, '--axis-diameters'),
    (begin_end_parser, '--axis-curvature'),    
    (axis_swath_parser, '--axis-swath'),
)

usage = "usage: %prog [options] contour_1 ... contour_n"

parser = optparse.OptionParser(usage=usage, description=__doc__.strip(), 
    add_help_option = False, formatter=cli_tools.CelltoolFormatter())
parser.set_defaults(
    show_progress=True,
    output_file='contour-measurements.csv'
)
parser.add_option('-h', '--help', action='store_true',
    help='print help text for all measurement commands and exit')
parser.add_option('-q', '--quiet', action='store_false', dest='show_progress',
    help='suppress progress bars and other status updates')
parser.add_option('-a', '--area', action='callback', callback=append_const, 
    callback_args=((simple_interface.Area, {}),), dest='measurements',
    help='measure the area of the contours in contour-units squared')
parser.add_option('-r', '--aspect-ratio', action='callback', callback=append_const,
    callback_args=((simple_interface.AspectRatio, {}),), dest='measurements',
    help='measure the aspect ratio (width / length) of the contours')
parser.add_option('-g', '--alignment-angle', action='callback', callback=append_const,
    callback_args=((simple_interface.AlignmentAngle, {}),), dest='measurements',
    help='measure the angle by which the original contour was rotated to bring it into alignment')
parser.add_option('-n', '--centroid', action='callback', callback=append_const,
    callback_args=((simple_interface.Centroid, {}),), dest='measurements',
    help='measure the (x, y) centroid of the each contour in the original image (in pixels)')
parser.add_option('-z', '--size', action='callback', callback=append_const,
    callback_args=((simple_interface.Size, {}),), dest='measurements',
    help='measure the (x, y) size of the each contour')
parser.add_option('-l', '--path-length', action='callback', callback=begin_end_handler,
    callback_args = (simple_interface.PathLength,), dest='measurements', type=None,
    help="measure the length of the contour perimeter [requires '-'-terminated arguments]")
parser.add_option('-c', '--curvature', action='callback', callback=begin_end_handler,
    callback_args = (simple_interface.NormalizedCurvature,), dest='measurements', type=None,
    help="measure the length-normalized curvature of the contours [requires '-'-terminated arguments]")
parser.add_option('-s', '--shape-modes', action='callback', callback=shape_mode_handler, 
    dest='measurements', type=None,
    help="measure the contours along one or more shape modes [requires '-'-terminated arguments]")
parser.add_option('-i', '--image-swath', action='callback', callback=image_swath_handler,
    dest='measurements', type=None,
    help="measure intensities of images along swaths defined by the contour outlines [requires '-'-terminated arguments]")
parser.add_option('-t', '--integrate-image', action='callback', callback=image_integral_handler,
    dest='measurements', type=None,
    help="measure integrated intensities of images within contour outlines [requires '-'-terminated arguments]")
parser.add_option('--axis-rmsd', action='callback', callback=append_const,
    callback_args=((simple_interface.AxisRMSD, {}),), dest='measurements',
    help='measure the RMSD of the central axis of the contour')
parser.add_option('--axis-relative-rmsd', action='callback', callback=append_const,
    callback_args=((simple_interface.RelativeAxisRMSD, {}),), dest='measurements',
    help='measure the RMSD of the central axis of the contour, relative to the baseline length')
parser.add_option('--axis-length-ratio', action='callback', callback=append_const,
    callback_args=((simple_interface.AxisLengthRatio, {}),), dest='measurements',
    help='measure the length of the central axis of the contour divided by the baseline length')
parser.add_option('--axis-peak-amplitude', action='callback', callback=append_const,
    callback_args=((simple_interface.AxisPeakAmplitude, {}),), dest='measurements',
    help='measure the maximum distance from the central axis of the contour to its baseline')
parser.add_option('--axis-wavelength', action='callback', callback=append_const,
    callback_args=((simple_interface.AxisWavelength, {}),), dest='measurements',
    help='estimate the wavelength of the central axis of the contour')
parser.add_option('--axis-wavenumber', action='callback', callback=append_const,
    callback_args=((simple_interface.AxisWavenumber, {}),), dest='measurements',
    help='estimate the number of cycles of the central axis of the contour')
parser.add_option('--axis-length', action='callback', callback=begin_end_handler,
    callback_args = (simple_interface.AxisLength,), dest='measurements', type=None,
    help="measure the length of the contour's central axis [requires '-'-terminated arguments]")
parser.add_option('--axis-mean-diameter', action='callback', callback=begin_end_handler,
    callback_args = (simple_interface.AxisMeanDiameter,), dest='measurements', type=None,
    help="measure the average diameter (width) of the contour along its central axis [requires '-'-terminated arguments]")
parser.add_option('--axis-diameters', action='callback', callback=begin_end_handler,
    callback_args = (simple_interface.AxisDiameters,), dest='measurements', type=None,
    help="measure the diameters (widths) of the contour along its central axis [requires '-'-terminated arguments]")
parser.add_option('--axis-curvature', action='callback', callback=begin_end_handler,
    callback_args = (simple_interface.AxisNormalizedCurvature,), dest='measurements', type=None,
    help="measure the normalized curvature of the contour's central axis [requires '-'-terminated arguments]")
parser.add_option('--axis-swath', action='callback', callback=axis_swath_handler,
    dest='measurements', type=None,
    help="measure intensities of images along swaths defined by the contour central axes [requires '-'-terminated arguments]")
parser.add_option('--side-curvature', action='callback', callback=two_span_handler,
    callback_args=(simple_interface.SideCurvature,), dest='measurements',
    type=int, nargs=4, metavar='begin_1 end_1 begin_2 end_2',
    help='sum the normalized curvatures along two regions of the contour')
parser.add_option('-o', '--output-file', metavar='FILE',
    help='name of data file to be written [default: %default]')

def main(name, arguments):
    parser.prog = name
    options, args = parser.parse_args(arguments)
    args = cli_tools.glob_args(args)
    if options.help:
        print_help()
    if len(args) == 0:
        raise ValueError('Some contour files must be specified!')
    contours = simple_interface.load_contours(args, show_progress = options.show_progress)
    measurements = [m(**kws) for m, kws in options.measurements]
    header, rows = simple_interface.measure_contours(contours, options.show_progress, *measurements)
    datafile.write_data_file([header]+rows, options.output_file)

def print_help():
    parser.print_help()
    print("\nMeasurements requiring option values terminated by a '-' (or just a '-'):\n")
    for sub_parser, name in sub_parsers:
        sub_parser.prog = name
        sub_parser.print_help()
        print('\n')
    parser.exit(2)

if __name__ == '__main__':
    import sys
    import os
    main(os.path.basename(sys.argv[0]), sys.argv[1:])