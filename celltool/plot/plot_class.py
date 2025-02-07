# Copyright 2007 Zachary Pincus
# This file is part of CellTool.
#
# CellTool is free software; you can redistribute it and/or modify
# it under the terms of version 2 of the GNU General Public License as
# published by the Free Software Foundation.

from . import svg_draw
import numpy
import copy

_GENERIC_DECENDER = 0.0
_GENERIC_BASELINE = 0.25
_GENERIC_X_HEIGHT = 0.75
_GENERIC_ASCENDER = 1.0
_GENERIC_CAP_CENTER = (_GENERIC_ASCENDER + _GENERIC_BASELINE) / 2.0
_GENERIC_LOW_CENTER = (_GENERIC_X_HEIGHT + _GENERIC_BASELINE) / 2.0
_TOP_TO_BASELINE = 1 - _GENERIC_BASELINE
_TOP_TO_CAP_CENTER = 1 - _GENERIC_CAP_CENTER
_TOP_TO_LOW_CENTER = 1 - _GENERIC_LOW_CENTER
_CAP_CENTER_TO_BASELINE = _GENERIC_CAP_CENTER - _GENERIC_BASELINE
_LOW_CENTER_TO_BASELINE = _GENERIC_LOW_CENTER - _GENERIC_BASELINE
_COMFORTABLE_PAD = 0.25

_PLOT_BOUNDS = False

class Style(object):
    def __init__(self, other = None):
        self.selectors = {}
        if other is not None:
            self.selectors = copy.deepcopy(other.selectors)
    def add_selector(self, selector, **attributes):
        selector = self.selectors.setdefault(selector, {})
        selector.update(dict([(k.replace('_', '-'), v) for k, v in attributes.items()]))
    def clear_selector(self, selector):
        self.selectors[selector] = {}
    def to_css(self):
        lines = []
        for selector, attributes in sorted(self.selectors.items()):
            lines.append('%s {'%selector)
            for k, v in sorted(attributes.items()):
                lines.append('\t%s: %s;'%(k, v))
            lines.append('}')
        return 'text/css', '\n'.join(lines)

default_style = Style()
default_style.add_selector('line.axis', stroke='dimgrey', stroke_width='0.75')
default_style.add_selector('line.tic', stroke_linecap='round')
default_style.add_selector('text', font_family='Times')
default_style.add_selector('text.label', font_weight='normal')
default_style.add_selector('text.title', font_weight='bold')
default_style.add_selector('text.axis', fill='dimgrey')
default_style.add_selector('line.scalebar', stroke='black', stroke_width='0.75')
default_style.add_selector('rect.plot', stroke='none', fill='none')
default_style.add_selector('.data', stroke='black', fill='none', stroke_width='1.5', stroke_linecap='round')
default_style.add_selector('line.colorbar.tic', stroke_width='0.5', stroke='black', stroke_linecap='butt')
default_style.add_selector('text.colorbar.label', fill='black')
default_style.add_selector('rect.colorbar', stroke_width='0.5', stroke='black', fill='none')

class Plot(object):
    def __init__(self, canvas_width, canvas_height, data_xrange, data_yrange, equal_axis_spacing = False,
            left_pad = 0, right_pad = 0, top_pad = 0, bottom_pad = 0, style=default_style):
        self.style = Style(other = style)
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.defs = []
        self.layers = {}
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        world_xrange = numpy.array([left_pad, canvas_width - left_pad - right_pad], dtype=float)
        world_yrange = numpy.array([top_pad, canvas_height - top_pad - bottom_pad], dtype=float)
        world_width = numpy.ptp(world_xrange)
        world_height = numpy.ptp(world_yrange)
        if equal_axis_spacing:
            data_xrange, data_yrange = _equal_aspect_ranges(data_xrange, data_yrange, world_width / float(world_height))
        self.data_xrange = data_xrange
        self.data_yrange = data_yrange
        data_width = numpy.ptp(numpy.asarray(data_xrange, dtype=float))
        data_height = numpy.ptp(numpy.asarray(data_yrange, dtype=float))
        self.data_to_world_scaling = numpy.array([world_width / data_width, world_height / data_height])
        self.world_min = numpy.array([world_xrange[0], world_yrange[1]])
        self.data_min = numpy.array([data_xrange[0], data_yrange[0]])
        if _PLOT_BOUNDS:
            self.make_layer('plot bounds', -10)
            plot = svg_draw.rect(world_xrange[0], world_yrange[0], world_width, world_height, id='plot bounds', svg_class='plot')
            self.add_to_layer('plot bounds', plot)

    def data_to_world_coordinates(self, points):
        points = numpy.asarray(points, dtype=float)
        return (points - self.data_min) * self.data_to_world_scaling * [1, -1] + self.world_min

    def make_layer(self, name, weight = None):
        """Make a new named layer, or just re-weight the old one if it already exists."""
        try:
            w, order, g = self.layers[name]
            if weight is None or w == weight: return
        except:
            order = len(self.layers)
            g = svg_draw.group(id=name, svg_class='layer')
            if weight is None: weight = 0
        self.layers[name] = (weight, order, g)
        return g

    def add_to_layer(self, name, element):
        if name not in self.layers:
            self.make_layer(name)
        weight, order, g = self.layers[name]
        g.addElement(element)

    def to_svg(self, filename = None, title = None):
        type, css = self.style.to_css()
        svg_style = svg_draw.style(type, cdata=css)
        defs = svg_draw.defs()
        if title is not None:
            svg_title = svg_draw.title(title)
            defs.addElement(svg_title)
        defs.addElement(svg_style)
        for element in self.defs:
            defs.addElement(element)
        self.svg = svg_draw.svg([0,0,self.canvas_width, self.canvas_height], self.canvas_width, self.canvas_height)
        self.svg.addElement(defs)
        for weight, number, group in sorted(self.layers.values()):
            self.svg.addElement(group)
        d = svg_draw.drawing()
        d.setSVG(self.svg)
        d.toXml(filename)

    def add_axis(self, axis, range, offset, tics, labels, smalltics, ticsize, smalltic_factor, font_size, title):
        if axis == 'x':
            axis_vector = numpy.array([1,0])
            tics_vector = numpy.array([0,1])
            text_anchor = 'middle'
            text_offset = numpy.array([0, ticsize/2.0 + font_size * (_TOP_TO_BASELINE + _COMFORTABLE_PAD)])
        elif axis == 'y':
            axis_vector = numpy.array([0,1])
            tics_vector = numpy.array([-1,0])
            text_anchor = 'end'
            text_offset = numpy.array([-font_size * _COMFORTABLE_PAD, font_size * _CAP_CENTER_TO_BASELINE])
        else:
            raise ValueError("Axis parameter must be 'x' or 'y'.")
        if labels == None:
            labels = [None] * len(tics)
        halftic = ticsize / 2.0
        offset = offset * numpy.flipud(axis_vector)
        start, stop = numpy.multiply.outer(range, axis_vector) + offset
        (x1, y1), (x2, y2) = self.data_to_world_coordinates([start, stop])
        axis_name = '%s-axis'%axis
        g = svg_draw.group(id=axis_name)
        self.add_to_layer('axes', g)
        g.addElement(svg_draw.line(x1, y1, x2, y2, id=axis_name, svg_class='axis'))
        tg = svg_draw.group(id = axis_name+' tics')
        lg = svg_draw.group(id = axis_name+' labels')
        for t, l in zip(tics, labels):
            position = self.data_to_world_coordinates(axis_vector * t + offset)
            x1, y1 = position - halftic * tics_vector
            x2, y2 = position + halftic * tics_vector
            tg.addElement(svg_draw.line(x1, y1, x2, y2, id='tic at %g'%t, svg_class='axis tic'))
            if l:
                x, y = position + halftic * tics_vector + text_offset
                lg.addElement(svg_draw.text(x, y, l, font_size=font_size, text_anchor=text_anchor, svg_class='axis label'))
        if len(lg.elements) > 0:
            self.add_to_layer('axes', lg)
        if len(tg.elements) > 0:
            g.addElement(tg)
        sg = svg_draw.group(id = axis_name+' small tics')
        for t in smalltics:
            position = self.data_to_world_coordinates(axis_vector * t + offset)
            x1, y1 = position - halftic * smalltic_factor * tics_vector
            x2, y2 = position + halftic * smalltic_factor * tics_vector
            sg.addElement(svg_draw.line(x1, y1, x2, y2, id='small tic at %g'%t, svg_class='axis tic small'))
        if len(sg.elements) > 0:
            g.addElement(sg)
        if title is not None:
            position = self.data_to_world_coordinates(axis_vector * range[1] + offset)
            x, y = position - (halftic + font_size * (_GENERIC_BASELINE)) * tics_vector
            if axis == 'x':
                t = svg_draw.text(x, y, title, font_size=font_size, text_anchor='middle', svg_class='axis title')
            else:
                t = svg_draw.text(x, y, title, font_size=font_size, text_anchor='start', transform='rotate(90, %g, %g)'%(x, y), svg_class='axis title')
            self.add_to_layer('axis titles', t)

    def add_axes(self, tics = (None, None), positions = (0, 0), titles = (None, None), ticsize = 6, font_size = 12, smalltics = 1):
        xtics, ytics = tics
        x_position, y_position = positions
        xtitle, ytitle = titles
        valid_positions = (-1, 0, 1)
        if x_position not in valid_positions or y_position not in valid_positions:
            raise ValueError("Valid axis positions are -1, 1, and 0.")
        xlow, xhigh = self.data_xrange
        ylow, yhigh = self.data_yrange
        if x_position == 0:
            x_true_position = numpy.clip(0, ylow, yhigh)
        elif x_position == -1:
            x_true_position = ylow
        else:
            x_true_position = yhigh
        if y_position == 0:
            y_true_position = numpy.clip(0, xlow, xhigh)
        elif y_position == -1:
            y_true_position = xlow
        else:
            y_true_position = xhigh
        if xtics is None:
            other_position = None
            if x_position != -1:
                # if the x-axis isn't at the bottom, make sure there are no tics where
                # the axes cross
                other_position = y_true_position
            xtics, xlabels, xsmalltics = _make_tics(xlow, xhigh, other_position, None, smalltics)
        else:
            xtics, xlabels, xsmalltics = xtics
        if ytics is None:
            other_position = None
            if y_position != -1:
                # if the y-axis isn't at the left, make sure there are no tics where
                # the axes cross
                other_position = x_true_position
            ytics, ylabels, ysmalltics = _make_tics(ylow, yhigh, other_position, None, smalltics)
        else:
            ytics, ylabels, ysmalltics = ytics
        self.make_layer('axes', -2)
        if titles[0] is not None and titles[1] is not None:
            self.make_layer('axis titles', -1)
        self.add_axis('x', self.data_xrange, x_true_position, xtics, xlabels, xsmalltics, ticsize, 0.5, font_size, xtitle)
        self.add_axis('y', self.data_yrange, y_true_position, ytics, ylabels, ysmalltics, ticsize, 0.5, font_size, ytitle)

    def add_bezier(self, points, id, svg_class='data', layer='data', in_data_coords=True):
        path = self._bezier_to_path(points, id, svg_class, in_data_coords)
        self.add_to_layer(layer, path)
        return path

    def add_polyline(self, points, id, svg_class='data', layer='data', in_data_coords=True):
        path = self._polyline_to_path(points, id, svg_class, in_data_coords)
        self.add_to_layer(layer, path)
        return path

    def add_circle(self, position, radius, id, svg_class='data', layer='data', in_data_coords=True):
        if in_data_coords:
            position = self.data_to_world_coordinates(position)
            radius *= self.data_to_world_scaling[0]
        circle = svg_draw.circle(cx=position[0], cy=position[1], r=radius, id=id, svg_class=svg_class)
        self.add_to_layer(layer, circle)
        return circle

    def _bezier_to_path(self, points, id, svg_class, in_data_coords):
        first_point = points[0]
        if in_data_coords:
            first_point = self.data_to_world_coordinates(first_point)
        pathdata = svg_draw.pathdata(x=first_point[0,0], y=first_point[0,1])
        for point in points:
            if in_data_coords:
                point = self.data_to_world_coordinates(point)
            if len(point) == 4:
                pathdata.bezier(*point[1:].ravel())
            elif len(point) == 3:
                pathdata.qbezier(*point[1:].ravel())
            elif len(point) == 2:
                pathdata.line(*point[1:].ravel())
        if numpy.all(points[0][0] == points[-1][-1]):
            pathdata.closepath()
        path = svg_draw.path(pathdata, id=id, svg_class=svg_class)
        return path

    def _polyline_to_path(self, points, id, svg_class, in_data_coords):
        points = numpy.asarray(points)
        first_point = points[0]
        if in_data_coords:
            first_point = self.data_to_world_coordinates(first_point)
        pathdata = svg_draw.pathdata(x=first_point[0], y=first_point[1])
        for point in points[1:]:
            if in_data_coords:
                point = self.data_to_world_coordinates(point)
            pathdata.line(x=point[0], y=point[1])
        if numpy.all(points[0] == points[-1]):
            pathdata.closepath()
        path = svg_draw.path(pathdata, id=id, svg_class=svg_class)
        return path


    def add_scalebar(self, x, y, length, ticheight, text, font_size):
        self.make_layer('scale bar', 1)
        g = svg_draw.group(id='scalebar lines')
        self.add_to_layer('scale bar', g)
        halftic = ticheight / 2.
        g.addElement(svg_draw.line(x, y, x+length, y, id='scalebar', svg_class='scalebar'))
        g.addElement(svg_draw.line(x, y-halftic, x, y+halftic, id='scalebar left tic', svg_class='scalebar tic'))
        g.addElement(svg_draw.line(x+length, y-halftic, x+length, y+halftic, id='scalebar right tic', svg_class='scalebar tic'))
        t = svg_draw.text(x+length/2., y+halftic+font_size*(_TOP_TO_BASELINE+_COMFORTABLE_PAD), text, font_size=font_size, text_anchor='middle', svg_class='scalebar label')
        self.add_to_layer('scale bar', t)

    def add_legend(self, x, y, width, names, font_size, ids = None, svg_classes = None, box=False):
        current_y = y
        if ids is None:
            ids = names
        if svg_classes is None:
            svg_classes = ['data legend']*len(names)
        self.make_layer('legend', 1)
        for svg_class, id, name in zip(svg_classes, ids, names):
            g = svg_draw.group(id='%s row'%name)
            if box:
                height = font_size * 0.9
                l = svg_draw.rect(x, current_y-height/2.0, width, height, id=id, svg_class=svg_class)
            else:
                l = svg_draw.line(x, current_y, x+width, current_y, id=id, svg_class=svg_class)
            text_y = current_y + font_size * (_LOW_CENTER_TO_BASELINE)
            text_x = x + width + font_size * _COMFORTABLE_PAD
            t = svg_draw.text(text_x, text_y, name, font_size=font_size, text_anchor='start', svg_class='legend label')
            current_y += font_size * (_COMFORTABLE_PAD + 1)
            g.elements.extend([l,t])
            self.add_to_layer('legend', g)

    def add_colorbar(self, x, y, width, height, range, font_size, gradient, range_namer = lambda x: str(x)):
        range = numpy.asarray(range)
        colorbar_layer = self.make_layer('color bar', 10)
        self.defs.append(gradient)
        self.style.add_selector('rect.colorbar', fill='url(#%s)'%gradient.attributes['id'])
        colorbar = svg_draw.rect(x, y, width, height, id='colorbar', svg_class='colorbar')
        colorbar_layer.addElement(colorbar)
        tics, labels, smalltics = _make_tics(range[0], range[1], num_smalltics = 1)
        if smalltics[-1] == range[1]:
            smalltics = smalltics[:-1]
        if tics[-1] != range[1]:
            tics = list(tics)
            tics.append(range[1])
        tg = svg_draw.group(id='colorbar tics')
        lg = svg_draw.group(id='colorbar labels')
        colorbar_layer.elements.extend([lg, tg])
        tic_length = 4
        tic_range = (range[1]-range[0])
        for tic in tics:
            tic_y = height * tic / tic_range    + y
            label = range_namer(tic)
            t = svg_draw.line(x, tic_y, x - tic_length, tic_y, id='colorbar tic at %s'%label, svg_class='colorbar tic')
            tg.addElement(t)
            label_x = x - tic_length - font_size * _COMFORTABLE_PAD
            label_y = tic_y + font_size * _CAP_CENTER_TO_BASELINE
            l = svg_draw.text(label_x, label_y, label, text_anchor='end', svg_class='colorbar label')
            lg.addElement(l)
        for tic in smalltics:
            tic_y = height * tic / tic_range + y
            label = range_namer(tic)
            t = svg_draw.line(x, tic_y, x - tic_length/2.0, tic_y, id='colorbar smalltic at %s'%label, svg_class='colorbar tic')
            tg.addElement(t)

    def add_title(self, title, font_size):
        x = self.canvas_width / 2.0
        y = font_size * (_TOP_TO_BASELINE + _COMFORTABLE_PAD)
        t = svg_draw.text(x, y, title, font_size=font_size, text_anchor='middle', svg_class='plot title')
        self.make_layer('title', 10)
        self.add_to_layer('title', t)



def _make_tics(low, high, other_axis_position = None, interval = None, num_smalltics = 0):
    if interval is None:
        interval = _find_tic_interval(low, high)
    original_low, original_high = low, high
    if low <= 0 and high >= 0 and other_axis_position == 0:
        if high%interval == 0:
            high += interval / 2.0
        if low%interval == 0:
            low -= interval / 2.0
        high_tics = numpy.arange(0, high, interval)
        low_tics = numpy.arange(0, low, -interval)
        tics = numpy.concatenate([numpy.flipud(low_tics), high_tics])
    else:
        quotient, remainder = divmod(low, interval)
        if remainder == 0:
            start = low
        else:
            start = (quotient + 1) * interval
        if (high - start)%interval == 0:
            high += interval / 2.0
        tics = numpy.arange(start, high, interval)
    labels = []
    new_tics = []
    for tic in tics:
        if other_axis_position is None or tic != other_axis_position:
            new_tics.append(tic)
            labels.append('%g'%tic)
    smalltics = []
    if num_smalltics > 0:
        # first segment
        st = numpy.linspace(tics[0] - interval, tics[0], num_smalltics+1, False)
        st = st.compress(st >= original_low)
        smalltics.extend(st)
        # middle segments
        for tic in tics[:-1]:
            st = numpy.linspace(tic, tic + interval, num_smalltics+1, False)[1:]
            smalltics.extend(st.compress(st <= original_high))
        # last segment
        st = numpy.linspace(tics[-1] + interval, tics[-1], num_smalltics+1, False)
        st = numpy.flipud(st)
        smalltics.extend(st.compress(st <= original_high))
    return new_tics, labels, smalltics

def _find_tic_interval(low, high):
    data_range = float(high) - low
    # We'll choose from between 3 and 8 tick marks
    divisions = numpy.arange(3, 9, dtype=float)
    candidate_intervals = data_range / divisions
    magnitudes = 10.0**numpy.floor(numpy.log10(candidate_intervals))
    mantissas = candidate_intervals / magnitudes
    # 'pleasing' intervals between tic-mark mantissas
    magic_intervals = numpy.array([1,2,2.5,5,10], dtype=float)
    min_difference = numpy.inf
    for magnitude, mantissa in zip(magnitudes, mantissas):
        difference = numpy.abs(mantissa - magic_intervals)
        # note equals below: give preference to more tics
        if difference.min() <= min_difference:
            best_magic = magic_intervals[difference.argmin()]
            best_magnitude = magnitude
    result = best_magic*best_magnitude
    if result == 0.0:
        result = limits.float_epsilon
    return result

def _equal_aspect_ranges(data_xrange, data_yrange, target_aspect):
    data_xrange, data_yrange = numpy.asarray(data_xrange, dtype=float), numpy.asarray(data_yrange, dtype=float)
    data_width, data_height = numpy.ptp(data_xrange), numpy.ptp(data_yrange)
    data_aspect = data_width / data_height
    if data_aspect > target_aspect:
        # need to expand the y-range to decrease the aspect ratio
        new_height = data_width / target_aspect
        diff = new_height - data_height
        data_yrange += [-diff/2, diff/2]
    elif data_aspect < target_aspect:
        # need to expand the x-range to increase the aspect ratio
        new_width = data_height * target_aspect
        diff = new_width - data_width
        data_xrange += [-diff/2, diff/2]
    return data_xrange, data_yrange
