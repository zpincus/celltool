# Copyright 2007 Zachary Pincus
# This file is part of CellTool.
#
# CellTool is free software; you can redistribute it and/or modify
# it under the terms of version 2 of the GNU General Public License as
# published by the Free Software Foundation.
import itertools
import bisect
import numpy
from scipy.stats import kde

import plot_class
import svg_draw
from celltool.contour import contour_class
from celltool.numerics import utility_tools
from celltool.utility import warn_tools
from celltool.utility import path
from celltool.utility.terminal_tools import progress_list, ProgressBar

old_default_colors = ['firebrick', 'green', 'cornflowerblue', 'orchid', 'darkslategray',
        'darkorange', 'lawngreen', 'midnightblue', 'lightgray', 'gold']

default_colors = ['rgb(67, 0, 246)', 'rgb(255, 147, 0)', 'rgb(31, 234, 181)',
    'rgb(255, 20, 247)', 'rgb(84, 13, 60)', 'rgb(49, 238, 231)', 'rgb(0, 71, 69)',
    'rgb(186, 31, 0)']

class GradientFactory(object):
    def __init__(self, stops = None):
        self.stops = []
        self.colors = []
        if stops is not None:
            for stop in stops:
                self.add_stop(*stop)
    def add_stop(self, percent, rgb):
        index = bisect.bisect(self.stops, percent)
        self.stops.insert(index, percent)
        self.colors.insert(index, numpy.array(rgb))
    def color_at(self, percent):
        if percent <= self.stops[0]:
            return self.colors[0]
        if percent >= self.stops[-1]:
            return self.colors[-1]
        interval = bisect.bisect(self.stops, percent)
        low, high = self.stops[interval-1:interval+1]
        clow, chigh = self.colors[interval-1:interval+1]
        fraction = (percent - low) / float(high - low)
        return chigh * fraction + clow * (1 - fraction)
    def svg_gradient(self, name, orientation = 'horizontal'):
        if orientation == 'horizontal':
            vector = (0, 100, 0, 0)
        elif orientation == 'vertical':
            vector = (0, 0, 0, 100)
        else:
            raise ValueError("Orientation must be either 'horizontal' or 'vertical'.")
        x1, x2, y1, y2 = ['%d%%'%v for v in vector]
        g = svg_draw.lineargradient(x1, y1, x2, y2, id=name)
        for p, rgb in zip(self.stops, self.colors):
            g.addElement(svg_draw.stop('%d%%'%p, _color_filter(rgb)))
        return g

default_gradient = GradientFactory([(0, (0,0,0)), (30, (70, 0, 255)), (70, (255, 70, 0)), (100, (255, 150, 0))])

def scatterplot(data_groups, filename, axis_titles = (None, None), scale = 0.005,
        plot_title = None, colors = default_colors, names = None, axes_at_origin = True,
        fix_xrange = (None, None), fix_yrange = (None, None)):
    """Plot circles at given x, y locations to an SVG file.

    The resulting SVG will have numerical axes (either centered at the origin, or
    placed on the left and bottom of the plot) and optionally a legend.

    Parameters:
        data_groups -- a list of groups to be plotted, where each group will be
            placed in a different SVG group and colored differently. Each data
            group is itself a list of [x, y] pairs.
        filename -- file to write the SVG out to.
        scale -- Diameter of the circles to be plotted, in units of the figure's
            horizontal width. (E.g. a value of 0.01 means 100 circles would fit
            along the x-axis.)
        axis_titles -- pair of titles for the x and y axes, respectively.
        plot_title -- title for the plot.
        colors -- the colors to fill the contours of each group with. Either color
            names that are defined in the SVG specification or (r,g,b) triplets are
            acceptable. This parameter must be an iterable; if there are not enough
            elements for the groups, they will be cycled.
        names -- the name of each group. Must be a list as long as contour_groups.
            If names is None, then there will be no legend in the output SVG.
        axes_at_origin -- if True, then the plot axes will intersect at the origin
            (or the nearest point to it in the data plot). Otherwise the axes will
            be at the bottom and left of the plot.
        fix_xrange, fix_yrange -- (min, max) tuples. If either or both values are
            None, then the best-fit range given the contours and their positions is
            chosen. Setting either or both forces the axis to have that minimum or
            maximum point.
    """

    data_groups = [[p for p in dg if _in_range(p, fix_xrange, fix_yrange)] for dg in data_groups]
    all_points = numpy.array([p for dg in data_groups for p in dg])

    plot_width = _CANVAS_WIDTH - _LEFT_PAD - _RIGHT_PAD
    plot_height = _CANVAS_HEIGHT - _TOP_PAD - _BOTTOM_PAD
    data_xrange, data_yrange = numpy.transpose([all_points.min(axis=0), all_points.max(axis=0)])
    fmin, fmax = fix_xrange
    if fmin is not None: data_xrange[0] = fmin
    if fmax is not None: data_xrange[1] = fmax
    fmin, fmax = fix_yrange
    if fmin is not None: data_yrange[0] = fmin
    if fmax is not None: data_yrange[1] = fmax
    equal_axis_spacing = False
    radius = scale * (data_xrange[1] - data_xrange[0])
    plot = plot_class.Plot(_CANVAS_WIDTH, _CANVAS_HEIGHT, data_xrange, data_yrange,
        equal_axis_spacing, _LEFT_PAD, _RIGHT_PAD, _TOP_PAD, _BOTTOM_PAD)
    legend = True
    if names is None:
        legend = False
        names = ['group-%d'%d for d in range(len(data_groups))]
    else:
        names = [name.replace(' ', '_') for name in names]
    plot.style.add_selector('.data', stroke='none')
    svg_classes = []
    for name, data_group, color in zip(names, data_groups, itertools.cycle(colors)):
        svg_class='data %s'%name
        svg_classes.append(svg_class)
        for point in data_group:
            plot.add_circle(point, radius, id=None, svg_class=svg_class, layer=name, in_data_coords=True)
        plot.style.add_selector('[class~="data"][class~="%s"]'%name, fill=color)
    if legend:
        legend_x = _CANVAS_WIDTH - _PAD - 80
        legend_y = _FONT_SIZE_SMALL * plot_class._TOP_TO_BASELINE + _PAD
        line_length = 50
        plot.add_legend(legend_x, legend_y, line_length, names, _FONT_SIZE_SMALL, svg_classes=svg_classes, box=True)

    if plot_title is not None:
        plot.add_title(plot_title, _FONT_SIZE)
    if axes_at_origin:
        positions = (0,0)
    else:
        positions = (-1,-1)
    plot.add_axes(positions = positions, titles = axis_titles, ticsize = _TICSIZE, font_size = _FONT_SIZE_SMALL)
    plot.to_svg(filename, plot_title)
    return plot

def line_plot(data_groups, filename, axis_titles = (None, None), plot_title = None,
        colors = default_colors, names = None, axes_at_origin = True,
        fix_xrange = (None, None), fix_yrange = (None, None), bezier=None):
    """Plot smooth lines connecting at given x, y locations to an SVG file.

    The resulting SVG will have numerical axes (either centered at the origin, or
    placed on the left and bottom of the plot) and optionally a legend.

    Parameters:
        data_groups -- a list of polylines to be plotted, where each polyline is
            placed in a different SVG group and colored differently. Each polyline
            is itself a list of [x, y] pairs.
        filename -- file to write the SVG out to.
        line_width -- Pixel width of the lines to be plotted.
        axis_titles -- pair of titles for the x and y axes, respectively.
        plot_title -- title for the plot.
        colors -- the colors to fill the contours of each group with. Either color
            names that are defined in the SVG specification or (r,g,b) triplets are
            acceptable. This parameter must be an iterable; if there are not enough
            elements for the groups, they will be cycled.
        names -- the name of each group. Must be a list as long as contour_groups.
            If names is None, then there will be no legend in the output SVG.
        axes_at_origin -- if True, then the plot axes will intersect at the origin
            (or the nearest point to it in the data plot). Otherwise the axes will
            be at the bottom and left of the plot.
        fix_xrange, fix_yrange -- (min, max) tuples. If either or both values are
            None, then the best-fit range given the contours and their positions is
            chosen. Setting either or both forces the axis to have that minimum or
            maximum point.
    """
    from scipy.interpolate import fitpack
    all_points = numpy.array([p for dg in data_groups for p in dg])

    plot_width = _CANVAS_WIDTH - _LEFT_PAD - _RIGHT_PAD
    plot_height = _CANVAS_HEIGHT - _TOP_PAD - _BOTTOM_PAD
    data_xrange, data_yrange = numpy.transpose([all_points.min(axis=0), all_points.max(axis=0)])
    fmin, fmax = fix_xrange
    if fmin is not None: data_xrange[0] = fmin
    if fmax is not None: data_xrange[1] = fmax
    fmin, fmax = fix_yrange
    if fmin is not None: data_yrange[0] = fmin
    if fmax is not None: data_yrange[1] = fmax
    equal_axis_spacing = False
    plot = plot_class.Plot(_CANVAS_WIDTH, _CANVAS_HEIGHT, data_xrange, data_yrange,
        equal_axis_spacing, _LEFT_PAD, _RIGHT_PAD, _TOP_PAD, _BOTTOM_PAD)
    legend = True
    if names is None:
        legend = False
        names = ['group-%d'%d for d in range(len(data_groups))]
    else:
        names = [name.replace(' ', '_') for name in names]
    for name, data_group, color in zip(names, data_groups, itertools.cycle(colors)):
        if bezier is not None:
            spline = fitpack.splprep(numpy.transpose(data_group), s=bezier)[0]
            curve = utility_tools.b_spline_to_bezier_series(spline)
            plot.add_bezier(curve, id=name, svg_class='data line', layer=name)
        else:
            plot.add_polyline(data_group, id=name, svg_class='data line', layer=name)
        plot.style.add_selector('[id="%s"]'%name, fill='none', stroke=color)
    if legend:
        legend_x = _CANVAS_WIDTH - _PAD - 80
        legend_y = _FONT_SIZE_SMALL * plot_class._TOP_TO_BASELINE + _PAD
        line_length = 50
        plot.add_legend(legend_x, legend_y, line_length, names, _FONT_SIZE_SMALL, box=False)
    if plot_title is not None:
        plot.add_title(plot_title, _FONT_SIZE)
    if axes_at_origin:
        positions = (0,0)
    else:
        positions = (-1,-1)
    plot.add_axes(positions = positions, titles = axis_titles, ticsize = _TICSIZE, font_size = _FONT_SIZE_SMALL)
    plot.to_svg(filename, plot_title)
    return plot


def contour_scatterplot(contour_groups, filename, scale = None, axis_titles = (None, None),
        plot_title = None, colors = default_colors, names = None, scalebar = True,
        axes_at_origin = True, fix_xrange = (None, None), fix_yrange = (None, None),
        show_contour_axes = True, show_progress = False):
    """Plot contours shapes at given x, y locations to an SVG file.

    The resulting SVG will have numerical axes (either centered at the origin, or
    placed on the left and bottom of the plot), a scale bar, and optionally a legend.

    Parameters:
        contour_groups -- a list of groups to be plotted, where each group will be
            placed in a different SVG group and colored differently. Each contour
            group is itself a list of (contour_object, [x, y]) tuples, where [x, y]
            is the point where the contour will be plotted.
        filename -- file to write the SVG out to.
        scale -- scaling factor to be multiplied to the contours before plotting.
            The scale is taken into account when creating the scale bar. If None,
            then a default scale will be chosen which scales the shapes so that
            about 25 of them laid end-to-end could fit across the plot axes.
        axis_titles -- pair of titles for the x and y axes, respectively.
        plot_title -- title for the plot.
        colors -- the colors to fill the contours of each group with. Either color
            names that are defined in the SVG specification or (r,g,b) triplets are
            acceptable. This parameter must be an iterable; if there are not enough
            elements for the groups, they will be cycled.
        names -- the name of each group. Must be a list as long as contour_groups.
            If names is None, then there will be no legend in the output SVG.
        scalebar -- if True, add a scale bar to the plot (denominated in the units
            that the contours are scaled in).
        axes_at_origin -- if True, then the plot axes will intersect at the origin
            (or the nearest point to it in the data plot). Otherwise the axes will
            be at the bottom and left of the plot.
        fix_xrange, fix_yrange -- (min, max) tuples. If either or both values are
            None, then the best-fit range given the contours and their positions is
            chosen. Setting either or both forces the axis to have that minimum or
            maximum point.
        show_contour_axes -- If any of the contours have a central axis defined
            and show_contour_axes is true, these will be plotted on the scatterplot.
    """
    contours = []
    points = []
    recentered_groups = []
    for contour_group in contour_groups:
        recentered_contours = []
        for c, p in contour_group:
            if not _in_range(p, fix_xrange, fix_yrange):
                continue
            r = c.as_recentered()
            recentered_contours.append((r, p))
            contours.append(r)
            points.append(p)
        recentered_groups.append(recentered_contours)
    contour_groups = recentered_groups
    units = _check_units(contours)
    plot_width = _CANVAS_WIDTH - _LEFT_PAD - _RIGHT_PAD
    plot_height = _CANVAS_HEIGHT - _TOP_PAD - _BOTTOM_PAD
    scale, data_xrange, data_yrange = _fit_contours(contours, points, plot_width, plot_height, scale, fix_xrange, fix_yrange)
    equal_axis_spacing = False
    plot = plot_class.Plot(_CANVAS_WIDTH, _CANVAS_HEIGHT, data_xrange, data_yrange,
        equal_axis_spacing, _LEFT_PAD, _RIGHT_PAD, _TOP_PAD, _BOTTOM_PAD)

    legend = True
    if names is None:
        legend = False
        names = ['group-%d'%d for d in range(len(contour_groups))]
    else:
        names = [name.replace(' ', '_') for name in names]
    plot.style.add_selector('.data', stroke='black', stroke_width='0.5')
    axis = show_contour_axes and _have_axis(contours)
    if axis:
        plot.style.add_selector('.data.contour.axis', stroke='white', stroke_width='0.5')
    svg_classes = []
    if show_progress:
        progress = ProgressBar('Plotting Contours')
        total = sum([len(cg) for cg in contour_groups])
        i = 0
    flip_y = numpy.array([[1,0,0],[0,-1,0],[0,0,1]])
    for name, contour_group, color in zip(names, contour_groups, itertools.cycle(colors)):
        svg_class='data contour %s'%name
        svg_classes.append(svg_class)
        for contour, point in contour_group:
            if show_progress:
                i += 1
                progress.update(float(i)/total, contour.simple_name())
            contour.scale(scale)
            contour.transform(flip_y)
            world_point = plot.data_to_world_coordinates(point)
            contour.recenter(world_point)

            ###
            simple_name = contour.simple_name()
            if axis:
                ap = _get_axis_points(plot, contour, simple_name, False, 1, in_data_coords=False)
            else:
                ap = None
            if ap is not None:
                g = svg_draw.group(id=simple_name)
                p = plot._bezier_to_path(contour.to_bezier(), id=simple_name, svg_class=svg_class, in_data_coords=False)
                g.addElement(p)
                g.addElement(ap)
                plot.add_to_layer(name, g)
            else:
                plot.add_bezier(contour.to_bezier(), id=simple_name, svg_class=svg_class, layer=name, in_data_coords=False)

            # plot.add_bezier(contour.to_bezier(), id=contour.simple_name(), svg_class=svg_class, layer=name, in_data_coords=False)
        plot.style.add_selector('[class~="data"][class~="%s"]'%name, fill=color)
    if legend:
        legend_x = _CANVAS_WIDTH - _PAD - 80
        legend_y = _FONT_SIZE_SMALL * plot_class._TOP_TO_BASELINE + _PAD
        line_length = 50
        plot.add_legend(legend_x, legend_y, line_length, names, _FONT_SIZE_SMALL, svg_classes=svg_classes, box=True)

    if plot_title is not None:
        plot.add_title(plot_title, _FONT_SIZE)
    if axes_at_origin:
        positions = (0,0)
    else:
        positions = (-1,-1)
    plot.add_axes(positions = positions, titles = axis_titles, ticsize = _TICSIZE, font_size = _FONT_SIZE_SMALL)
    if scalebar:
        _add_scalebar(plot, units, scale)
    plot.to_svg(filename, plot_title)
    return plot

def _in_range(p, xrange, yrange):
    x, y = p
    xmin, xmax = xrange
    ymin, ymax = yrange
    return (xmin is None or x >= xmin) and (xmax is None or x <= xmax) and (ymin is None or y >= ymin) and (ymax is None or y <= ymax)

def _fit_contours(contours, points, plot_width, plot_height, scale, fix_xrange = (None, None), fix_yrange = (None, None)):
    points = numpy.asarray(points, dtype=float)
    bounds = numpy.array([points.min(axis=0), points.max(axis=0)])
    contour_sizes = numpy.array([contour.size() for contour in contours])
    biggest = contour_sizes.max(axis = 0)
    if scale is None:
        scale = numpy.min(numpy.array([plot_width, plot_height]) / (biggest * 25))
        print('NOTE: automatically-chosen scale value was %g.'%scale)
    else:
        max_scale = numpy.min(numpy.array([plot_width, plot_height]) / biggest)
        if scale > max_scale:
            raise ValueError('The maximum scale value is %g; otherwise some shape(s) will be too large for the plot area. (%g was supplied).'%(max_scale, scale))
    if fix_xrange[0] is not None and fix_yrange[0] is not None and fix_xrange[1] is not None and fix_yrange[1] is not None:
        return scale, fix_xrange, fix_yrange
    contour_halves = contour_sizes * scale / 2 + 4
    # +4 is to allow a few pixels of padding on each side.
    data_xrange, data_yrange = bounds.transpose()
    # Now iteratively fit the scaled contours into the range. The first iteration
    # we will adjust the ranges enough to fit the contours perfectly given the
    # old range. However, it will slightly undershoot given the *new* range.
    # We'll optimize for a while to stabilize...
    iters = 0
    while True:
        if iters > 25:
            warn_tools.warn('Could not fit the given shapes to the given plot bounds. There will be some overlap.')
            break
        data_width, data_height = data_xrange.ptp(), data_yrange.ptp()
        scaling = numpy.array([plot_width / data_width, plot_height / data_height])
        world_points = (points - [data_xrange[0], data_yrange[0]]) * scaling
        max_points = (world_points + contour_halves).max(axis = 0)
        min_points = (world_points - contour_halves).min(axis = 0)
        max_fudge = (max_points - [plot_width, plot_height]) / scaling
        min_fudge = (min_points) / scaling
        if numpy.allclose(max_fudge, 0) and numpy.allclose(min_fudge, 0):
            break
        data_xrange += [min_fudge[0], max_fudge[0]]
        data_yrange += [min_fudge[1], max_fudge[1]]
        # If any of the fix_range values are not None, clamp our values to them
        if fix_xrange[0] is not None:
            data_xrange[0] = fix_xrange[0]
        if fix_xrange[1] is not None:
            data_xrange[1] = fix_xrange[1]
        if fix_yrange[0] is not None:
            data_yrange[0] = fix_yrange[0]
        if fix_yrange[1] is not None:
            data_yrange[1] = fix_yrange[1]
        iters += 1
    return scale, data_xrange, data_yrange

def distribution_plot(data_groups, filename, axis_title = None, plot_title = None,
        colors = default_colors, names = None, axes_at_origin = True, plot_points = False,
 	fix_xrange = (None, None), scale_factors = None):
    """Plot the 1D distributions of several collections of points to a SVG file.

    The distributions of the 1D points in 'data_groups' are estimated with kernel
    density estimation, fit to splines, and plotted on numerical axes with an
    optional legend.

    Parameters:
        data_groups -- a list of grouped data points. Each group is a list of numbers,
            the distribution of which will be estimated and plotted.
        filename -- file to write the SVG out to.
        axis_title -- Title for the x-axis.
        plot_title -- title for the plot.
        colors -- the colors to fill the contours of each group with. Either color
            names that are defined in the SVG specification or (r,g,b) triplets are
            acceptable. This parameter must be an iterable; if there are not enough
            elements for the groups, they will be cycled.
        names -- the name of each group. Must be a list as long as contour_groups.
            If names is None, then there will be no legend in the output SVG.
        axes_at_origin -- if True, then the plot axes will intersect at the origin
            (or the nearest point to it in the data plot). Otherwise the axes will
            be at the bottom and left of the plot.
	    plot_points -- if True, dots will be plotted at the location of each
	        sample.
        fix_xrange, fix_yrange -- (min, max) tuples. If either or both values are
            None, then the best-fit range given the contours and their positions is
            chosen. Setting either or both forces the axis to have that minimum or
            maximum point.
        scale_factors -- If not None, a list of values by which to scale the heights
            of each distribution.
    """
    from scipy.interpolate import fitpack
    data_groups = [numpy.asarray(data_group, dtype=numpy.float32) for data_group in data_groups]
    if not numpy.alltrue([data_group.ndim == 1 for data_group in data_groups]):
        raise ValueError('Can only plot distributions in one dimension.')
    data_groups = [numpy.asarray(data_group) for data_group in data_groups]
    data_ranges = [(g.min(), g.max()) for g in data_groups]
    kd_estimators = [kde.gaussian_kde(data_group) for data_group in data_groups]
    min = numpy.min([m for m, x in data_ranges])
    max = numpy.max([x for m, x in data_ranges])
    extra = 0.05*(max - min)
    min -= extra
    max += extra
    #min, max, data_ranges = _kde_range(kd_estimators, data_ranges, 5e-6)
    # pin the values to those of fix_xrange, if specified
    if fix_xrange[0] is not None:
        min = fix_xrange[0]
    if fix_xrange[1] is not None:
        max = fix_xrange[1]

    height = 0
    beziers = []
    err = numpy.seterr(under='ignore')
    # ignore underflow -- KDE can underflow when estimating regions of very low density
    if scale_factors is None:
        scale_factors = [1]*len(kd_estimators)
    for (range_min, range_max), estimator, scale_factor in zip(data_ranges, kd_estimators, scale_factors):
        if fix_xrange[0] is not None and range_min < min: range_min=min
        if fix_xrange[1] is not None and range_max > max: range_max=max
        eval_points = list(numpy.linspace(range_min, range_max, 100, True))
        kde_points = estimator.evaluate(eval_points) * scale_factor
        data_height = kde_points.max()
        if data_height > height: height = data_height
        #s = 0.0005 * numpy.min([range_max - range_min, data_height]) / 100
        s = 0
        spline = fitpack.splprep([eval_points, kde_points], u=eval_points, s=s)[0]
        beziers.append(utility_tools.b_spline_to_bezier_series(spline))
    numpy.seterr(**err)

    data_xrange = [min, max]
    data_yrange = [0, height]
    equal_axis_spacing = False
    plot = plot_class.Plot(_CANVAS_WIDTH, _CANVAS_HEIGHT, data_xrange, data_yrange,
        equal_axis_spacing, _LEFT_PAD, _RIGHT_PAD, _TOP_PAD, _BOTTOM_PAD)
    legend = True
    if names is None:
        legend = False
        names = ['distribution-%d'%d for d in range(len(data_groups))]
    else:
        names = [name.replace(' ', '_') for name in names]
    if plot_points:
    	point_radius = 0.005 * (data_xrange[1] - data_xrange[0])
    	plot.style.add_selector('[class~="point"]', fill='black', stroke=None)
    for name, bezier, data_group, estimator, color in zip(names, beziers, data_groups, kd_estimators, itertools.cycle(colors)):
        plot.add_bezier(bezier, id=name, svg_class='data density', layer=name)
        plot.style.add_selector('[class~="density"][id="%s"]'%name, fill='none', stroke=color)
        plot.style.add_selector('[class~="legend"][id="%s"]'%name, fill='none', stroke=color)
        if plot_points:
        	values = numpy.unique(data_group)
        	for point in zip(values, estimator(values)):
        	    plot.add_circle(point, point_radius, id=None, svg_class="data point", layer=name, in_data_coords=True)
    if legend:
        legend_x = _CANVAS_WIDTH - _PAD - 80
        legend_y = _FONT_SIZE_SMALL * plot_class._TOP_TO_BASELINE + _PAD
        line_length = 50
        plot.add_legend(legend_x, legend_y, line_length, names, _FONT_SIZE_SMALL)
    if plot_title is not None:
        plot.add_title(plot_title, _FONT_SIZE)
    if axes_at_origin:
        positions = (-1,0)
    else:
        positions = (-1,-1)
    plot.add_axes(positions = positions, titles = (axis_title, None), ticsize = _TICSIZE, font_size = _FONT_SIZE_SMALL)
    plot.to_svg(filename, plot_title)
    return plot

class _kde_height_list(object):
    def __init__(self, estimator, low, high, steps):
        self.estimator = estimator
        self.low = low
        self.high = high
        self.width = high - low
        self.steps = steps
    def __len__(self):
        return self.steps
    def __getitem__(self, i):
        return float(self.estimator(self.position_at_index(i)))
    def position_at_index(self, i):
        return self.width * float(i)/(self.steps-1) + self.low

def _kde_range(kd_estimators, data_ranges, zero_threshold):
    err = numpy.seterr(under='ignore')
    # ignore underflow -- KDE can underflow when estimating regions of very low density
    mins, maxes = [], []
    for e, (min, max) in zip(kd_estimators, data_ranges):
        range = (max - min)*2
        min_hl = _kde_height_list(e, min-range, min, 1024)
        mins.append(min_hl.position_at_index(bisect.bisect(min_hl, zero_threshold)))
        max_hl = _kde_height_list(e, max+range, max, 1024)
        maxes.append(max_hl.position_at_index(bisect.bisect(max_hl, zero_threshold)))
    numpy.seterr(**err)
    return numpy.min(mins), numpy.max(maxes), zip(mins, maxes)

def pca_modes_plot(pca_contour, filename, modes = None, positions = None,
        gradient_factory = default_gradient, scalebar = True, scale = None):
    """Plot the shape modes of a PCAContour object to an SVG file.

    The shape modes of a PCAContour object are plotted, left-to-right. For each
    mode, the shapes generated at various positions along the mode are superimposed
    on the plot. By default, these are shapes at -2sd, -1sd, the mean, 1sd and 2sd
    (where sd is the standard deviation of the data along that mode).

    Parameters:
        pca_contour -- a PCAContour object containing information about the shape modes.
        modes -- a list of the modes to plot (where a value of 1, not 0, indicates
         the first shape mode). If None, then all of the modes stored in pca_contour
         are used.
        filename -- file to write the SVG out to. If None, the SVG is returned as
         a string.
        positions -- the positions along the mode that will be plotted in
            superimposition. If None, then the positions at -2sd, -1sd, mean, 1sd and
            2sd are plotted.
        gradient_factory -- GradientFactory object to supply the color gradient
            that will be used to color the positions.
        scalebar -- if True, add a scale bar to the plot (denominated in the units
            that the contours are scaled in).
        scale -- output-pixels-per-contour-unit scale factor.
    """
    num_total_modes = len(pca_contour.modes)
    if not modes:
        modes = numpy.arange(num_total_modes)
    else:
        modes = numpy.asarray(modes) - 1
        for mode in modes:
            if mode < 0 or mode >= len(pca_contour.modes):
                raise ValueError("The PCA contour does not have shape mode '%d'."%(mode+1))
    if not positions:
        positions = numpy.array([-2, -1, 0, 1, 2])
    else:
        positions = numpy.asarray(positions)
    mode_contours = _get_mode_contours(pca_contour, modes, positions)
    bounds = [_find_bounds(contours) for contours in mode_contours]
    sizes = numpy.array([b.ptp(axis = 0) for b in bounds])
    width = sizes[:,0].sum()
    height = sizes[:,1].max()
    if len(modes) > 1:
        width_pad = 0.15*width
        width += width_pad
        individual_pad = width_pad / (len(modes) - 1)
    else:
        individual_pad = 0
    equal_axis_spacing = True
    data_xrange = [0, width]
    data_yrange = [-height/2.0, height/2.0]
    plot = plot_class.Plot(_CANVAS_WIDTH, _CANVAS_HEIGHT, data_xrange, data_yrange,
        equal_axis_spacing, _LEFT_PAD, _RIGHT_PAD, _TOP_PAD, _BOTTOM_PAD)
    if scale is None:
        print("Automatically-chosen scale is %g pixels per %s."%(plot.data_to_world_scaling[0], pca_contour.units))
    else:
        plot.data_to_world_scaling = numpy.array([scale, scale])
    current_x = 0
    for (w, h), bound, contours, mode in zip(sizes, bounds, mode_contours, modes):
        mode_name = mode + 1
        mode_text = 'Shape Mode %d'%mode_name
        mode_var = pca_contour.standard_deviations[mode]**2 / pca_contour.total_variance
        var_text = '%.1f'%(mode_var*100)
        if var_text == '0.0':
            var_text = '< 0.1'
        variance_text = ' (%s%% of total variance)'%var_text
        current_x += w / 2.0
        center_point = bound.mean(axis = 0)
        translation = numpy.array([current_x, 0]) - center_point
        for contour, pos in zip(contours, positions):
            contour.translate(translation)
            points = contour.to_bezier()
            plot.add_bezier(points, id='mode-%d position-%s'%(mode_name, pos), svg_class='data contour', layer=mode_text)
        text_x = plot.data_to_world_coordinates([current_x, 0])[0]
        text_y = _CANVAS_HEIGHT - _PAD - _FONT_SIZE * plot_class._GENERIC_BASELINE - _FONT_SIZE_SMALL
        t = svg_draw.text(text_x, text_y, mode_text, font_size=_FONT_SIZE, text_anchor='middle', svg_class='shapemode label')
        var_y = _CANVAS_HEIGHT - _PAD - _FONT_SIZE_SMALL * plot_class._GENERIC_BASELINE
        v = svg_draw.text(text_x, var_y, variance_text, font_size=_FONT_SIZE_SMALL, text_anchor='middle', svg_class='shapemode label')
        plot.add_to_layer(mode_text, t)
        plot.add_to_layer(mode_text, v)
        current_x += w / 2.0 + individual_pad

    legend_x = _CANVAS_WIDTH - _PAD - 80
    legend_y = _FONT_SIZE_SMALL * plot_class._TOP_TO_BASELINE + _PAD
    line_length = 50
    names, ids = [], []
    color_func = _gradient_color_namer([positions.min(), positions.max()], gradient_factory)
    for p in positions:
        ids.append('position-%s'%p)
        plot.style.add_selector('[id~="position-%s"]'%p, stroke=color_func(p))
        if p == 0:
            names.append(u'mean')
            #names.append(u'\N{GREEK SMALL LETTER MU}')
        else:
            names.append(u'%d s.d.'%p)
            #names.append(u'%d\N{GREEK SMALL LETTER SIGMA}'%p)
    if scalebar:
        _add_scalebar(plot, pca_contour.units)
    plot.add_legend(legend_x, legend_y, line_length, names, _FONT_SIZE_SMALL, ids)
    plot_title = contour.simple_name()
    plot.add_title(plot_title, _FONT_SIZE)
    plot.to_svg(filename, plot_title)
    return plot

def _get_mode_contours(pca_contour, modes, positions):
    num_total_modes = len(pca_contour.modes)
    mode_contours = []
    for mode in modes:
        inner_contours = []
        for position in positions:
            pos = numpy.zeros(num_total_modes)
            pos[mode] = position
            inner_contours.append(pca_contour.as_position(pos, normalized = True))
        mode_contours.append(inner_contours)
    return mode_contours

def contour_plot(contours, filename, plot_title = None, scalebar = True,
    gradient_factory = default_gradient, show_progress = False,
    radii = False, radius_steps = 1, scale = None):
    """Plot    one or more contours to an SVG file.

    Parameters:
        contours -- a list of Contour objects to plot.
        filename -- file to write the SVG out to.
        plot_title -- title for the plot.
        scalebar -- if True, add a scale bar to the plot (denominated in the units
            that the contours are scaled in).
        gradient_factory -- GradientFactory object to supply the color gradient
            for contour colorings. If none, do not color...
        radii and radius_steps are used for contours with central axes defined:
            if radii is True, then the every radius_steps along the axis, the radius
            will be plotted.
        scale -- output-pixels-per-contour-unit scale factor.
    """
    units = _check_units(contours)
    bounds = _find_bounds(contours)
    data_xrange, data_yrange = bounds.transpose()
    equal_axis_spacing = True
    plot = plot_class.Plot(_CANVAS_WIDTH, _CANVAS_HEIGHT, data_xrange, data_yrange,
        equal_axis_spacing, _LEFT_PAD, _RIGHT_PAD, _TOP_PAD, _BOTTOM_PAD)
    if scale is None:
        print("Automatically-chosen scale is %g pixels per %s."%(plot.data_to_world_scaling[0], units))
    else:
        plot.data_to_world_scaling = numpy.array([scale, scale])
    landmarks = _have_landmarks(contours)
    axis = _have_axis(contours)
    if gradient_factory is None:
        plot.style.add_selector('path.data.contour', stroke='black', stroke_width='2.5', fill='none')
        if landmarks:
            plot.style.add_selector('circle.data.landmark', fill='black', stroke='none')
        if axis:
            plot.style.add_selector('path.data.contour.axis', stroke='black', stroke_width='1', fill='none')
            if radii:
                plot.style.add_selector('polyline.data.contour.radius', stroke='black', stroke_width='1', fill='none')
    else:
        plot.style.add_selector('path.data.contour', stroke_width='2.5', fill='none')
        if landmarks:
            plot.style.add_selector('circle.data.landmark', stroke='none')
        if axis:
            plot.style.add_selector('path.data.contour.axis', stroke_width='1', fill='none')
            if radii:
                plot.style.add_selector('polyline.data.contour.radius', stroke_width='1', fill='none')
        if len(contours) > 1:
            color_func = _gradient_color_namer((0, len(contours)-1), gradient_factory)
        else:
            def color_func(*args):
                return 'black'
    if show_progress:
        econtours = progress_list(enumerate(contours), 'Plotting Contours', lambda ic: ic[1].simple_name())
    else:
        econtours = enumerate(contours)
    for i, contour in econtours:
        name = contour.simple_name().replace(' ', '_')
        if landmarks or axis:
            g = svg_draw.group(id=name)
            lm = _get_landmark_points(plot, contour, name)
            if lm is not None:
                g.addElement(lm)
            ap = _get_axis_points(plot, contour, name, radii, radius_steps)
            if ap is not None:
                g.addElement(ap)
            p = plot._bezier_to_path(contour.to_bezier(), id=name, svg_class='data contour', in_data_coords=True)
            g.addElement(p)
            plot.add_to_layer('contours', g)
        else:
            plot.add_bezier(contour.to_bezier(), id=name, svg_class='data contour', layer='contours')
        if gradient_factory is not None:
            plot.style.add_selector('[id~="%s"]'%name, fill='none', stroke=color_func(i))
            if landmarks:
                plot.style.add_selector('circle.data.landmark[id~="%s"]'%name, fill=color_func(i))
    if scalebar:
        _add_scalebar(plot, units)
    if plot_title is not None:
        plot.add_title(plot_title, _FONT_SIZE)
    plot.to_svg(filename, plot_title)
    return plot


def point_order_plot(contours, filename, plot_title = None, label_points = True,
        colorbar = True, begin = None, end = None, gradient_factory = default_gradient,
        scalebar = True, color_by_point = True, show_progress = False, scale=None):
    """Plot the order of points in one or more contours to an SVG file.

    One or more contours are plotted from point 'begin' to point 'end', optionally
    colored by their point ordering. Optionally some of the points are labeled,
    and/or a colorbar showing the ordering is provided.

    Parameters:
        contours -- a list of Contour objects to plot.
        filename -- file to write the SVG out to.
        plot_title -- title for the plot.
        label_points -- if True, points will be labeled at evenly-spaced intervals.
        colorbar -- if True, a colorbar will be added.
        begin -- if not None, only contour points after this index will be plotted.
        end -- if not None, the contour points before this index will be plotted.
        gradient_factory -- GradientFactory object to supply the color gradient.
        scalebar -- if True, add a scale bar to the plot (denominated in the units
            that the contours are scaled in).
        color_by_point -- if True, contours are colored by their points; if false
            by their order in the 'contours' list.
        scale -- output-pixels-per-contour-unit scale factor.
    """
    if not utility_tools.all_same_shape([c.points for c in contours]):
        raise ValueError('All contours must have the same number of points.')
    units = _check_units(contours)
    point_range = utility_tools.inclusive_periodic_slice(range(len(contours[0].points)), begin, end)
    num_points = len(point_range)
    contours = [contour.as_recentered() for contour in contours]
    bounds = _find_bounds(contours)
    data_xrange, data_yrange = bounds.transpose()
    colorbar_width = 30
    equal_axis_spacing = True
    plot = plot_class.Plot(_CANVAS_WIDTH, _CANVAS_HEIGHT, data_xrange, data_yrange,
        equal_axis_spacing, _LEFT_PAD, _RIGHT_PAD + colorbar_width, _TOP_PAD, _BOTTOM_PAD)
    if scale is None:
        print("Automatically-chosen scale is %g pixels per %s."%(plot.data_to_world_scaling[0], units))
    else:
        plot.data_to_world_scaling = numpy.array([scale, scale])
    if show_progress:
        contours_prog = progress_list(contours, 'Plotting Contours', lambda c: c.simple_name())
    else:
        contours_prog = contours
    plot.style.selectors['.data'].pop('stroke')
    landmarks = _have_landmarks(contours)
    if landmarks:
        plot.style.add_selector('circle.data.landmark', stroke='none')
    color_func = None
    if gradient_factory is not None:
        plot.style.add_selector('path.data.range', stroke_width='2.5', fill='none', stroke_linecap='round')
        if color_by_point:
            color_func = _gradient_color_namer([0, num_points-1], gradient_factory)
            if landmarks:
                plot.style.add_selector('circle.data.landmark', fill='darkgray')
        else:
            style_color_func = _gradient_color_namer((0, len(contours)-1), gradient_factory)
            for i, contour in enumerate(contours):
                plot.style.add_selector('[id="%s"]'%contour.simple_name(), stroke=style_color_func(i))
                if landmarks:
                    plot.style.add_selector('circle.data.landmark[id~="%s"]'%contour.simple_name(), fill=style_color_func(i))
    else:
        plot.style.add_selector('path.data.range', stroke_width='2.5', fill='none', stroke='black', stroke_linecap='round')
        if landmarks:
            plot.style.add_selector('circle.data.landmark', fill='black')
    svg_groups = [_gradient_contour(plot, contour, begin, end, range(num_points), color_func) for contour in contours_prog]
    if begin is not None or end is not None:
        plot.style.add_selector('path.data.fullpath', stroke_width='0.5', fill='none', stroke='darkgray')
    contour_layer = plot.make_layer('contours')
    for g in svg_groups:
        contour_layer.addElement(g)
    if gradient_factory is not None and colorbar:
        def namer(x):
            if int(x) != x: return ''
            else: return str(point_range[int(x)]+1)
        #namer = lambda x: str(point_range[int(x)]+1)
        _add_colorbar(plot, [0, len(point_range)-1], colorbar_width, gradient_factory, _FONT_SIZE_SMALL, namer)
    if scalebar:
        _add_scalebar(plot, units)
    if label_points:
        _add_point_labels(plot, contours, begin, end)
    if plot_title is not None:
        plot.add_title(plot_title, _FONT_SIZE)
    plot.to_svg(filename, plot_title)
    return plot

def _check_units(contours):
    units = [contour.units for contour in contours]
    if not numpy.alltrue([u == units[0] for u in units]):
        raise ValueError('All contours must have the same units in order to plot them on the same axes.')
    units = units[0]
    return units

def _find_bounds(contours):
    all_points = numpy.concatenate([contour.points for contour in contours])
    mins = all_points.min(axis = 0)
    maxes = all_points.max(axis = 0)
    return numpy.array([mins, maxes])

def _gradient_color_namer(range, gradient_factory):
    width = float(range[1] - range[0])
    def color_func(x):
        return _color_filter(gradient_factory.color_at(100 * (x - range[0]) / width))
    return color_func

def _gradient_contour(plot, contour, begin, end, values, color_func):
    curves = utility_tools.inclusive_periodic_slice(contour.to_bezier(match_curves_to_points = True), begin, end)
    if end is not None:
        # if an end-point was specified, make the last element END at that end-point,
        # and do not draw the arc from that point to the next one!
        curves = curves[:-1]
    name = contour.simple_name()
    g = svg_draw.group(id=name)
    lm = _get_landmark_points(plot, contour, name)
    if lm is not None:
        g.addElement(lm)
    if begin is not None or end is not None:
        p = plot._bezier_to_path(contour.to_bezier(), id=None, svg_class='data fullpath', in_data_coords=True)
        g.addElement(p)
    if color_func is None:
        p = plot._bezier_to_path(curves, id=contour.simple_name(), svg_class='data range', in_data_coords=True)
        g.addElement(p)
    else:
        for value, curve in zip(values, curves):
            d = svg_draw.pathdata(*plot.data_to_world_coordinates(curve[0]))
            d.bezier(*plot.data_to_world_coordinates(curve[1:]).ravel())
            p = svg_draw.path(d, stroke=color_func(value), fill='none', svg_class='data range')
            g.addElement(p)
    return g

def _have_landmarks(contours):
    for contour in contours:
        if isinstance(contour, contour_class.ContourAndLandmarks):
            return True
    return False

def _have_axis(contours):
    for contour in contours:
        if isinstance(contour, contour_class.CentralAxisContour):
            return True
    return False

def _get_landmark_points(plot, contour, name):
    if not isinstance(contour, contour_class.ContourAndLandmarks):
        return None
    g = svg_draw.group(id=contour.simple_name()+' landmarks')
    for i, point in enumerate(contour.landmarks):
        x, y = plot.data_to_world_coordinates(point)
        g.addElement(svg_draw.circle(x, y, 3, id=name+' %d'%i, svg_class='data landmark'))
    return g

def _get_axis_points(plot, contour, name, radii=False, radius_steps=1, in_data_coords=True):
    if not isinstance(contour, contour_class.CentralAxisContour):
        return None
    axis = plot._bezier_to_path(contour.axis_to_bezier(), id=name+' axis', svg_class='data contour axis', in_data_coords=in_data_coords)
    if radii:
        g = svg_draw.group(id=contour.simple_name()+' radii')
        g.addElement(axis)
        for i, (radius, normal, point) in enumerate(zip(contour.radii, contour.axis_normals(), contour.axis)):
            if i % radius_steps != 0:
                continue
            if in_data_coords:
                x, y = plot.data_to_world_coordinates(point)
                dx, dy = radius * normal * plot.data_to_world_scaling * [1, -1]
            else:
                x, y = point
                dx, dy = radius * normal
            g.addElement(svg_draw.polyline([[x-dx, y-dy], [x, y], [x+dx, y+dy]], id=name+' %d'%i, svg_class='data contour radius'))
        return g
    else:
        return axis

def _add_colorbar(plot, range, colorbar_width, gradient_factory, font_size, range_namer = lambda x: str(x)):
    gradient_name = 'colorbar_gradient'
    gradient = gradient_factory.svg_gradient(gradient_name, 'vertical')
    colorbar_height = plot.canvas_height / 2.0
    colorbar_y = colorbar_height / 2.0
    colorbar_x = plot.canvas_width - colorbar_width - 4
    plot.add_colorbar(colorbar_x, colorbar_y, colorbar_width, colorbar_height, range, font_size, gradient, range_namer)

def _pick_scale_length(scale_factor, target_length = 50):
    """Find a nice, round number that when multiplied by scale_factor gives a length
    close to target_length."""
    magic_numbers = numpy.array([1,2,2.5,5,10], dtype=float)
    scaled_target = target_length / scale_factor
    magnitude = 10.0**numpy.floor(numpy.log10(scaled_target))
    mantissa = scaled_target / magnitude
    best = numpy.absolute(magic_numbers - mantissa).argmin()
    return magic_numbers[best]*magnitude

def _add_scalebar(plot, units, scale = None, x=30, y=None):
    if y is None:
        y = plot.canvas_height
    if scale is None:
        scale_x, scale_y = plot.data_to_world_scaling
        if not numpy.allclose(scale_x, scale_y):
            raise RuntimeError("Plot does not have equal x- and y-scaling. Cannot create a valid scale bar.")
        scale = scale_x
    length = _pick_scale_length(scale)
    ticheight = 3
    if int(length) == length:
        text = '%d %s'%(length, units)
    else:
        text = '%g %s'%(length, units)
    scaled_length = length * scale
    plot.add_scalebar(x, y, scaled_length, ticheight, text, _FONT_SIZE_SMALL)

def _add_point_labels(plot, contours, begin, end):
    point_range = utility_tools.inclusive_periodic_slice(range(len(contours[0].points)), begin, end)
    num_points = len(point_range)
    index_tics, labels, smalltics = plot_class._make_tics(0, num_points - 1, num_smalltics = 0)
    if begin is not None and end is not None:
        if index_tics[-1] != num_points - 1:
            index_tics = list(index_tics)
            index_tics.append(num_points - 1)
    index_tics = numpy.asarray(index_tics, dtype=int)
    tics = numpy.take(point_range, index_tics)
    contour_points = [numpy.take(contour.points, tics, axis = 0) for contour in contours]
    points_at_tics = numpy.transpose(contour_points, (1,0,2))
    distances = [utility_tools.norm(points).argsort() for points in points_at_tics]
    sorted_points = [numpy.take(points, indices, axis = 0) for points, indices in zip(points_at_tics, distances)]
    farthest = [d[-1] for d in distances]
    outward_normals = []
    for f, index in zip(farthest, index_tics):
        contour = contours[f]
        first_der = contour.first_derivatives(begin, end)[index]
        outward_normal = numpy.array([first_der[1], -first_der[0]])
        if contour.signed_area() > 0:
            outward_normal *= -1
        outward_normals.append(outward_normal)
    plot.style.add_selector('polyline.data.legend.connector', stroke_width='0.75', stroke='black')
    cg = svg_draw.group(id='point label connectors')
    lg = svg_draw.group(id='point label text')
    layer = plot.make_layer('Point Labels', 20)
    layer.elements.extend([cg, lg])
    for tic, points, normal in zip(tics, sorted_points, outward_normals):
        final_length = 18
        text_length = final_length + 0.6 * _FONT_SIZE
        world_points = plot.data_to_world_coordinates(points)
        world_normal = normal * plot.data_to_world_scaling * [1, -1]
        world_normal /= numpy.sqrt((world_normal**2).sum())
        world_points = list(world_points)
        world_points.append(world_points[-1] + world_normal * final_length)
        c = svg_draw.polyline(list(world_points), id='point_connector %d'%tic, svg_class='data legend connector')
        cg.addElement(c)
        # if numpy.absolute(numpy.arctan2(normal[1],normal[0])) < numpy.pi/2:
        #     text_anchor = 'start'
        # else:
        #     text_anchor = 'end'
        text_x, text_y = world_points[-2] + world_normal * text_length
        text_y += plot_class._CAP_CENTER_TO_BASELINE*_FONT_SIZE
        l = svg_draw.text(text_x, text_y, str(tic+1), text_anchor='middle', font_size=_FONT_SIZE, svg_class='legend label')
        lg.addElement(l)


_FONT_SIZE = 14
_FONT_SIZE_SMALL = 12
_CANVAS_WIDTH = 640
_CANVAS_HEIGHT = 480
_PAD = 10
_LEFT_PAD = _FONT_SIZE + _PAD
_RIGHT_PAD = _PAD
_TOP_PAD = _FONT_SIZE + _PAD
_BOTTOM_PAD = _FONT_SIZE + _PAD
_TICSIZE = 6

def _color_filter(color):
    """Return a string representation of the color.

    Try to convert convert a (r,g,b) tuple to a color string, or just return
    the string version of the color parameter.
    """
    try:
        r,g,b = [(int(e)) for e in color]
        return 'rgb(%d,%d,%d)'%(r,g,b)
    except:
        return str(color)
