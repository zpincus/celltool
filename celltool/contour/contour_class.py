# Copyright 2007 Zachary Pincus
# This file is part of CellTool.
#
# CellTool is free software; you can redistribute it and/or modify
# it under the terms of version 2 of the GNU General Public License as
# published by the Free Software Foundation.

"""Classes for dealing with data organized as 2D point clouds and contours.

This module provides a base class, PointSet, for simple operations on sets of
2D points. A subclass, Contour, provides specified operations for points
organized into closed contours (aka polygons). Other subclasses deal with
contour data that also includes outside landmark points (ContourAndLandmarks),
or for creating and using PCA-based shape models (PCAContour).
"""

import numpy
import copy

from celltool.numerics import utility_tools
from celltool.numerics import procustes
from celltool.utility import path

_pi_over_2 = numpy.pi / 2

class ContourError(RuntimeError):
    pass

def _copymethod(method):
    """Return a function that makes a copy of the self object applys the method to that object, and returns the modified copy."""
    def m(self, *p, **kw):
        c = self.__class__(other = self)
        method(c, *p, **kw)
        return c
    m.__doc__ = 'Make a copy of this object, apply method %s (with the given arguments) to the copy, and then return the modified copy.' % method.__name__
    return m

class PointSet(object):
    """Manage a list of 2D points and provide basic methods to measure properties
    of those points and transform them geometrically.

    The list of points is stored in the 'points' variable, which is an array of
    size Nx2.

    Note that methods starting with 'as_' are equivalent to their similarly-named
    counterparts, except they first make a copy of the object, modify that copy,
    and return the new object. For example, q = p.as_recentered() is equivalent to:
    q = PointSet(other = p)
    q.recenter()
    """

    _instance_data = {'points' : numpy.zeros((0, 2)),
                                        'to_world_transform' : numpy.eye(3),
                                        'units' : None}

    def __init__(self, **kws):
        """Create a new PointSet (or subclass).

        The most important parameter is 'points', which must be convertible to an
        Nx2 array, specifying N data points in (x, y) format.

        Optional arguments to provide data for the new object are allowed; see the
        class's _instance_data attribute for paramter names and their default values.

        In addition, if an 'other' keyword is supplied, that parameter is assumed to
        be an other object of a compatible class, and the relevant attributes are
        copied (if possible) from the other object (AKA copy-construction). A dict
        can also be supplied for this parameter.

        If a keyword parameter and an attriubte from 'other' are both defined, the
        former has precedence. If neither are present, the default value from
        _instance_data is used.
        """
        try:
            other = kws['other']
        except:
            other = None
        for attr, value in self._instance_data.items():
            if attr in kws:
                value = kws[attr]
            elif other is not None:
                try:
                    value = getattr(other, attr)
                except:
                    try:
                        value = other[attr]
                    except:
                        pass
            if isinstance(self._instance_data[attr], numpy.ndarray):
                setattr(self, attr, numpy.array(value, copy=True, subok=True))
            else:
                setattr(self, attr, copy.deepcopy(value))
        self._filename = ''
        if other is not None:
            try:
                self._filename = other._filename
            except:
                pass

    def as_copy(self):
        """Return a copy of this object."""
        return self.__class__(other = self)

    def simple_name(self):
        """Return the base name (no directories, no extension) of the file that this
        object was loaded from."""
        try:
            return path.path(self._filename).namebase
        except:
            return ''
    def bounding_box(self):
        """Return the bounding box of the data points as [[xmin, ymin], [xmax, ymax]]."""
        mins = self.points.min(axis = 0)
        maxes = self.points.max(axis = 0)
        return numpy.array([mins, maxes])

    def size(self):
        """Return the size of the data point bounding box [x_size, y_size]"""
        return self.points.max(axis = 0) - self.points.min(axis = 0)

    def centroid(self):
        """Return the [x, y] centroid of the data points."""
        return self.points.mean(axis = 0)

    def alignment_angle(self):
        """Return the rotation (in degrees) needed to return the contour to its
        original alignment."""
        rotate_reflect = utility_tools.decompose_homogenous_transform(self.to_world_transform)[0]
        theta = numpy.arctan2(rotate_reflect[0,1], rotate_reflect[0,0])
        return theta * 180 / numpy.pi

    def bounds_center(self):
        """Return the center point of the bounding box. Differs from the centroid
        in that the centroid is weighted by the number of points in any particular
        location."""
        mins, maxes = self.bounding_box()
        return mins + (maxes - mins) / 2.0

    def aspect_ratio(self):
        """Return the aspect ratio of the data as x_size / y_size."""
        size = self.size()
        return float(size[0]) / size[1]

    def recenter(self, center = numpy.array([0,0])):
        """Center the data points about the provided center-point, or the origin if no point is provided."""
        center = numpy.asarray(center)
        self.translate(center - self.centroid())

    def recenter_bounds(self, center = numpy.array([0,0])):
        """Center the data points' bounding-box about the provided center-point, or the origin if no point is provided."""
        center = numpy.asarray(center)
        self.translate(center - self.bounds_center())

    def transform(self, transform):
        """Transform the data points with the provided affine transform.

        The transform should be a 3x3 transform in homogenous coordinates that will
        be used to transform row vectors by pre-multiplication.
        (E.g. final_point = transform * initial_point, where final_point and
        initial_point are 3x1, and * indicates matrix multiplication.)

        The provided transform is inverted and used to update the to_world_transform
        instance variable. This variable keeps track of all of the transforms performed
        so far.
        """
        inverse = numpy.linalg.inv(transform)
        self.to_world_transform = numpy.dot(inverse, self.to_world_transform)
        self.points = utility_tools.homogenous_transform_points(self.points, transform)

    def translate(self, translation):
        """Translate the points by the given [x,y] translation."""
        self.transform(utility_tools.make_homogenous_transform(translation = translation))

    def scale(self, scale):
        """Scale the points by the provide scaling factor (either a constant or an [x_scale, y_scale] pair)."""
        self.transform(utility_tools.make_homogenous_transform(scale = scale))

    def descale(self):
        """Remove any previously-applied scaling factors. If the contour is centered
        at the origin, it will remain so; if it is centered elsewhere then the descaling
        will be applied to its current location as well."""
        rotate_reflect, scale_shear, translation = utility_tools.decompose_homogenous_transform(self.to_world_transform)
        transform = utility_tools.make_homogenous_transform(transform=scale_shear)
        self.transform(transform)

    def rotate(self, rotation, in_radians = True):
        """Rotate the points by the given rotation, which can optionally be specified in degrees."""
        if not in_radians:
            rotation = numpy.pi * rotation / 180.
        s = numpy.sin(rotation)
        c = numpy.cos(rotation)
        self.transform(utility_tools.make_homogenous_transform(transform = [[c,s],[-s,c]]))

    def to_world(self):
        """Return the points to their original ('world') coordinates, undoing all transforms."""
        self.transform(self.to_world_transform)

    def to_file(self, filename):
        """Save the object to a named file.

        The saved file is valid python code which can be executed to re-create all of the
        object's instance variables."""
        old_threshold = numpy.get_printoptions()['threshold']
        numpy.set_printoptions(threshold = numpy.inf)
        file_contents = ['cls = ("%s", "%s")\n'%(self.__class__.__module__, self.__class__.__name__)]
        for var_name in self._instance_data:
            file_contents.append('%s = \\'%var_name)
            file_contents.append(repr(getattr(self, var_name, None)).replace('np.', ''))  # remove any np. prefixes since on load we use the numpy namespace
            file_contents.append('\n')
        file_contents = '\n'.join(file_contents)
        try:
            f = open(filename, 'w')
        except Exception as e:
            raise IOError('Could not open file "%s" for saving. (Error: %s)'%(filename, e))
        f.write(file_contents)
        f.close()
        numpy.set_printoptions(threshold = old_threshold)

    def rigid_align(self, reference, weights = None, allow_reflection = False, allow_scaling = False, allow_translation = True):
        """Find the best rigid alignment between the data points and those of another PointSet (or subclass) object.

        By default, the alignment can include translation and rotation; reflections
        or scaling transforms can also be allowed, or translation disallowed.
        In addition, the 'weights' parameter allows different weights to be set for
        each data point.

        The best rigid alignment (least-mean-squared-distance between the data points
        and the reference points) is returned as a 3x3 homogenous transform matrix
        which operates on row-vectors.
        """
        T, c, t, new_A = procustes.procustes_alignment(self.points, reference.points, weights,
            allow_reflection, allow_scaling, allow_translation)
        self.transform(utility_tools.make_homogenous_transform(T, c, t))

    def axis_align(self):
        """Align the data points so that the major and minor axes of the best-fit ellpise are along the x and y axes, respectively."""
        self.recenter()
        u, s, vt = numpy.linalg.svd(self.points, full_matrices = 0)
        rotation = -numpy.arctan2(vt[0, 1],vt[0,0])
        # If we're rotating by more than pi/2 radians, just go the other direction.
        if rotation > _pi_over_2 or rotation < -_pi_over_2:
            rotation += numpy.pi
        self.rotate(rotation)

    def rms_distance_from(self, reference):
        """Calculate the RMSD between the data points and those of a reference object."""
        return numpy.sqrt(((self.points - reference.points)**2).mean())

    def procustes_distance_from(self, reference, apply_transform = True,
            weights = None, allow_reflection = False, allow_scaling = False, allow_translation = True):
        """Calculate the procustes distance between the data points and those of a reference object.

        The procustes distance is the RMSD between two point sets after the best rigid transform
        between the object (some of rotation/translation/reflection/scaling, depending on the
        parameters to this function) is taken into account. Weights for the individual points can
        also be specified. (See rigid_align for more details.)

        By default, the rigid transform is applied to the data points as a side-effect
        of calculating the procustes distance, though this can be disabled.
        """
        T, c, t, new_A = procustes.procustes_alignment(self.points, reference.points, weights,
            allow_reflection, allow_scaling, allow_translation)
        if apply_transform:
            self.transform(utility_tools.make_homogenous_transform(T, c, t))
        return numpy.sqrt(((new_A - reference.points)**2).mean())

    as_world = _copymethod(to_world)
    as_recentered = _copymethod(recenter)
    as_recentered_bounds = _copymethod(recenter_bounds)
    as_transformed = _copymethod(transform)
    as_translated = _copymethod(translate)
    as_rotated = _copymethod(rotate)
    as_scaled = _copymethod(scale)
    as_descaled = _copymethod(descale)
    as_rigid_aligned = _copymethod(rigid_align)
    as_axis_aligned = _copymethod(axis_align)


class Contour(PointSet):
    """Class for dealing with an ordered set of points that comprise a contour or polygon.

    This subclass of PointSet provides methods appropriate for closed contours.
    Internally, the contour is stored in the 'points' attribute as an Nx2 array
    of (x, y) points. Note that points[0] != points[-1]; that is, the contour is
    not explicitly closed, though this is implicitly assumed. The constructor will
    take care of any explicitly closed point data, if provided.

    Please also review the PointSet documentation for relevant details, especially
    pertaining to the __init__ method and method with 'as_' names.
    """
    _instance_data = dict(PointSet._instance_data)

    def __init__(self, **kws):
        PointSet.__init__(self, **kws)
        self._make_acyclic()

    def area(self):
        """Return the area inside of the contour."""
        return numpy.abs(self.signed_area())

    def signed_area(self):
        """Return the signed area inside of the contour.

        If the contour points wind counter-clockwise, the area is negative; otherwise
        it is positive."""
        xs = self.points[:,0]
        ys = self.points[:,1]
        y_forward = numpy.roll(ys, -1, axis = 0)
        y_backward = numpy.roll(ys, 1, axis = 0)
        return numpy.sum(xs * (y_backward - y_forward)) / 2.0

    def reverse_orientation(self):
        """Reverse the orientation of the contour from clockwise to counter-clockwise or vice-versa."""
        self.points = numpy.flipud(self.points)

    def point_range(self, begin = None, end = None):
        """Get a periodic slice of the contour points from begin to end, inclusive.

        If 'begin' is after 'end', then the slice wraps around."""
        return utility_tools.inclusive_periodic_slice(self.points, begin, end)

    def length(self, begin = None, end = None):
        """Calculate the length of the contour, optionally over only the periodic slice specified by 'begin' and 'end'."""
        return self.interpoint_distances(begin, end).sum()

    def cumulative_distances(self, begin = None, end = None):
        """Calculate the cumulative distances along the contour, optionally over only the periodic slice specified by 'begin' and 'end'."""
        interpoint_distances = self.interpoint_distances(begin, end)
        interpoint_distances[0] = 0
        return numpy.add.accumulate(interpoint_distances)

    def interpoint_distances(self, begin = None, end = None):
        """Calculate the distance from each point to the previous point, optionally over only the periodic slice specified by 'begin' and 'end'."""
        offsetcontour = numpy.roll(self.points, 1, axis = 0)
        return utility_tools.inclusive_periodic_slice(utility_tools.norm(self.points - offsetcontour, axis = 0), begin, end)

    def spline_derivatives(self, begin, end, derivatives=1):
        """Calculate derivative or derivatives of the contour using a spline fit,
        optionally over only the periodic slice specified by 'begin' and 'end'."""
        from scipy.interpolate import fitpack
        try:
            l = len(derivatives)
            unpack = False
        except:
            unpack = True
            derivatives = [derivatives]
        tck, uout = self.to_spline()
        points = utility_tools.inclusive_periodic_slice(list(range(len(self.points))), begin, end)
        ret = [numpy.transpose(fitpack.splev(points, tck, der=d)) for d in derivatives]
        if unpack:
            ret = ret[0]
        return ret

    def first_derivatives(self, begin = None, end = None):
        """Calculate the first derivatives of the contour, optionally over only the periodic slice specified by 'begin' and 'end'."""
        return self.spline_derivatives(begin, end, 1)

    def second_derivatives(self, begin = None, end = None):
        """Calculate the second derivatives of the contour, optionally over only the periodic slice specified by 'begin' and 'end'."""
        return self.spline_derivatives(begin, end, 2)

    def curvatures(self, begin = None, end = None):
        """Calculate the curvatures of the contour (1/r of the osculating circle at each point), optionally over only the periodic slice specified by 'begin' and 'end'."""
        d1, d2 = self.spline_derivatives(begin, end, [1,2])
        x1 = d1[:,0]
        y1 = d1[:,1]
        x2 = d2[:,0]
        y2 = d2[:,1]
        return (x1*y2 - y1*x2) / (x1**2 + y1**2)**(3./2)

    def normalized_curvature(self, begin = None, end = None):
        """Return the mean of the absolute values of the curvatures over the given
        range, times the path length along that range. For a circle, this equals
        the angle (in radians) swept out along the range. For less smooth shapes,
        the value is higher."""
        return numpy.absolute(self.curvatures(begin, end)).mean() * self.length(begin, end)

    def inward_normals(self, positions=None):
        """Return unit-vectors facing inwards at the points specified in the
        positions variable (of all points, if not specified). Note that fractional
        positions are acceptable, as these values are calculated via spline
        interpolation."""
        from scipy.interpolate import fitpack
        if positions is None:
            positions = numpy.arange(len(self.points))
        tck, uout = self.to_spline()
        points = numpy.transpose(fitpack.splev(positions, tck))
        first_der = numpy.transpose(fitpack.splev(positions, tck, 1))
        inward_normals = numpy.empty_like(first_der)
        inward_normals[:,0] = -first_der[:,1]
        inward_normals[:,1] = first_der[:,0]
        if self.signed_area() > 0:
            inward_normals *= -1
        inward_normals /= numpy.sqrt((inward_normals**2).sum(axis=1))[:, numpy.newaxis]
        return inward_normals

    def interpolate_points(self, positions):
        """Use spline interpolation to determine the spatial positions at the
        contour positions specified (fracitonal positions are thus acceptable)."""
        from scipy.interpolate import fitpack
        tck, uout = self.to_spline()
        return numpy.transpose(fitpack.splev(positions, tck))

#    def _make_cyclic(self):
#        if not self.is_cyclic():
#            self.points = numpy.resize(self.points, [self.points.shape[0] + 1, self.points.shape[1]])
#            self.points[-1] = self.points[0]

    def _make_acyclic(self):
        """If the contour is cyclic (last point == first point), strip the last point off."""
        if numpy.all(self.points[-1] == self.points[0]):
            self.points = numpy.resize(self.points, [self.points.shape[0] - 1, self.points.shape[1]])

    def offset_points(self, offset):
        """Offset the point ordering forward or backward.

        Example: if the points are offset by 1, then the old points[0] is now at points[1],
        the old points[-1] is at points[0], and so forth. This doesn't change the spatial
        position of the contour, but it changes how the points are numbered.
        """
        self.points = numpy.roll(self.points, offset, axis = 0)

    def to_spline(self, smoothing = 0, spacing_corrected = False):
        """Return the best-fit periodic parametric 3rd degree b-spline to the data points.

        The smoothing parameter is an upper-bound on the mean squared deviation between the
        data points and the points produced by the smoothed spline. By default it is 0,
        forcing an interpolating spline.

        The returned spline is valid over the parametric range [0, num_points+1], where
        the value at 0 is the same as the value at num_points+1. If spacing_corrected
        is True, then the intermediate points will be placed according to the physical
        spacing between them, which is useful in some situations like resampling.
        However, it means that the spline evalueated at position N will not necessarialy
        give the same point as contour.points[n], unless all of the points are exactly
        evenly spaced. Similarly, non-zero values of smoothing will disrupt this property
        as well.

        Two values are returned: tck and u. 'tck' is a tuple containing the spline c
        oefficients, the knots (two lists; x-knots and y-knots), and the degree of
        the spline. This 'tck' tuple can be used by the routines in scipy.interpolate.fitpack.
        'u' is a list of the parameter values corresponding to the points in the range.
        """
        from scipy.interpolate import fitpack
        # the fitpack smoothing parameter is an upper-bound on the TOTAL squared deviation;
        # ours is a bound on the MEAN squared deviation. Fix the mismatch:
        l = len(self.points)
        smoothing = smoothing * l
        if spacing_corrected:
            interpoint_distances = self.interpoint_distances()
            last_to_first = interpoint_distances[0]
            interpoint_distances[0] = 0
            cumulative_distances = numpy.add.accumulate(interpoint_distances)
            u = numpy.empty(l+1, dtype=float)
            u[:-1] = cumulative_distances
            u[-1] = cumulative_distances[-1] + last_to_first
            u *= l / u[-1]
        else:
            u = numpy.arange(0, l+1)
        points = numpy.resize(self.points, [l+1, 2])
        points[-1] = points[0]
        tck, uout = fitpack.splprep(x = points.transpose(), u = u, per = True, s = smoothing)
        return tck, uout

    def to_bezier(self, match_curves_to_points = False, smooth = True):
        """Convert the contour into a sequence of cubic Bezier curves.

        NOTE: There may be fewer Bezier curves than points in the contour, if the
        contour is sufficiently smooth. To ensure that each point interval in the
        contour corresponds to a returned curve, set 'match_curves_to_points' to
        True.

        Output:
            A list of cubic Bezier curves.
            Each Bezier curve is an array of shape (4,2); thus the curve includes the
            starting point, the two control points, and the endpoint.
        """
        from scipy.interpolate import fitpack
        if smooth:
            size = self.size().max()
            s = 0.00001 * size
        else:
            s = 0
        tck, u = self.to_spline(smoothing=s)
        if match_curves_to_points:
            to_insert = numpy.setdiff1d(u, numpy.unique(tck[0]))
            for i in to_insert:
                tck = fitpack.insert(i, tck, per = True)
        return utility_tools.b_spline_to_bezier_series(tck, per = True)


    def resample(self, num_points, smoothing = 0, max_iters = 500, min_rms_change = 1e-6, step_size = 0.2):
        """Resample the contour to the given number of points, which will be spaced as evenly as possible.

        Parameters:
            - smoothing: the smoothing parameter for the spline fit used in resampling. See the to_spline documentation.
            - max_iters: the resampled points are evenly-spaced via an iterative process. This is the maximum number of iterations.
            - min_rms_change: if the points change by this amount or less, cease iteration.
            - step_size: amount to adjust the point spacing by, in the range [0,1]. Values too small slow convergence, but
                values too large introduce ringing. 0.2-0.6 is a generally safe range.

        Returns the number of iterations and the final RMS change.
        """
        # cache functions in inner loop as local vars for faster lookup
        from scipy.interpolate.fitpack import splev
        norm, roll, clip, mean = utility_tools.norm, numpy.roll, numpy.clip, numpy.mean
        iters = 0
        ms_change = numpy.inf
        l = len(self.points)
        tck, u = self.to_spline(smoothing, spacing_corrected = True)
        positions = numpy.linspace(0, l+1, num_points, endpoint = False)
        min_ms_change = min_rms_change**2
        points = numpy.transpose(splev(positions, tck))
        while (iters < max_iters and ms_change > min_ms_change):
            forward_distances = norm(points - roll(points, -1, axis = 0))
            backward_distances = norm(points - roll(points, 1, axis = 0))
            arc_spans = (roll(positions, -1, axis = 0) - roll(positions, 1, axis = 0)) % (l+1)
            deltas = forward_distances - backward_distances
            units = arc_spans / (forward_distances + backward_distances)
            steps = step_size * deltas * units
            steps[0] = 0
            positions += steps
            positions = clip(positions, 0, l+1)
            iters += 1
            ms_change = mean((steps**2))
            points = numpy.transpose(splev(positions, tck))
        self.points = points
        return iters, numpy.sqrt(ms_change)

    def global_reorder_points(self, reference):
        """Find the point ordering that best aligns (in the RMSD sense) the data points to the reference object's points.

        The 'best ordering' is defined as the offset which produces the smallest
        RMSD between the data points and the corresponding reference points
        (when the data points are so offset; see the offset_points method for details).
        A global binary search strategy works well because the RMSD-as-a-function-of-offset
        landscape is smooth and sinusoidal.
        (I have not proven this, but it is empirically so for simple cases of both
        convex and concave contours. Perhaps this is not valid for self-overlapping
        polygons, however.)

        Returns the final RMSD between the points (as best ordered) and the reference points.
        """
        best_offset = 0
        step = self.points.shape[0] // 2
        while(True):
            d = self.as_offset_points(best_offset).rms_distance_from(reference)
            dp = self.as_offset_points(best_offset + 1).rms_distance_from(reference)
            dn = self.as_offset_points(best_offset - 1).rms_distance_from(reference)
            if d < dp and d < dn: break
            elif dp < dn: direction = 1
            else: direction = -1
            best_offset += direction * step
            if step > 2:
                step //= 2
            else:
                step = 1
        self.offset_points(best_offset)
        return d


    def _local_point_ordering_search(self, reference, distance_function, max_iters = None):
        """Find the point ordering that best aligns the data points to the reference object's points.

        The quality of the alignment is evaluated by the provided distance function.
        A local search strategy is employed which takes unit steps in the most
        promising direction until a distance minima is reached.

        Note that the distance function might have side-effects on this object. This
        is desirable in the case that we want to transform the points as a part
        of finding the distance (e.g. we're looking at procustes distances).

        Returns the final distance value between the data points and the reference points.
        """
        if max_iters is None:
            max_iters = len(self.points)
        d = distance_function(self, reference)
        pos = self.as_offset_points(1)
        neg = self.as_offset_points(-1)
        dp = distance_function(pos, reference)
        dn = distance_function(neg, reference)
        if d < dp and d < dn: return d
        elif dp < dn:
            contour = pos
            d = dp
            direction = 1
        else:
            contour = neg
            d = dn
            direction = -1
        iters = 0
        while iters < max_iters:
            iters += 1
            ctr = contour.as_offset_points(direction)
            dp = distance_function(ctr, reference)
            if dp > d:
                break
            else:
                contour = ctr
                d = dp
        # now copy the metadata from the best contour to self.
        self.__init__(other = contour)
        return d

    def local_reorder_points(self, reference, max_iters = None):
        """Find the point ordering that best aligns (in the RMSD sense) the data points to the reference object's points.

        The 'best ordering' is defined as the offset which produces the smallest
        RMSD between the data points and the corresponding reference points
        (when the data points are so offset; see the offset_points method for details).
        A local hill_climbing search strategy is used.

        This function will be slower than global_reorder_points unless the maxima
        is closer than log2(len(points)).

        Returns the final RMSD between the data points and the reference points.
        """
        return self._local_point_ordering_search(reference, self.__class__.rms_distance_from, max_iters)

    def local_best_alignment(self, reference, weights = None, allow_reflection = False,
            allow_scaling = True, allow_translation = True, max_iters = None):
        """Find the point ordering that best aligns (in the procustes distance sense) the data points to the reference object's points.

        The 'best ordering' is defined as the offset which produces the smallest
        procustes between the data points (when the points are so offset and then
        procustes aligned to the reference points; see offset_points, rigid_align, and
        procustes_distance_from for more details). A local hill_climbing search strategy is used.

        The 'weights', 'allow_reflection', 'allow_scaling', and 'allow_translation'
        parameters are equivalent to those from the rigid_align method; which see for
        details.

        Returns the final procustes distance between the data points and the reference points.
        """
        pdf = self.__class__.procustes_distance_from
        def find_distance(contour, reference):
            return pdf(contour, reference, True, weights, allow_reflection, allow_scaling, allow_translation)
        return self._local_point_ordering_search(reference, find_distance, max_iters)

    def global_best_alignment(self, reference, align_steps = 8, weights = None, allow_reflection = False,
            allow_scaling = True, allow_translation = True, allow_reversed_orientation = True, quick = False):
        """Perform a global search for the point ordering that allows the best rigid alignment between the data points and a reference.

        The 'align_steps' parameter controls the number of offsets that are
        initially examined. The contour will be offset 'align_steps' times, evenly
        spaced. If the 'quick' parameter is False (the default), a local search is
        performed at each step to find the closest maxima; if 'quick' is True then
        the distance at that offset (and not a nearby maxima) is recorded. In
        either case, one final local search is performed to refine the best fit
        previously found.

         The 'weights', 'allow_reflection', 'allow_scaling', and
        'allow_translation' parameters are equivalent to those from the
        rigid_align method; which see for details.

         If the 'allow_reversed_orientation' parameter is true, than at each step,
        the contour ordering is reversed to see if that provides a better fit.
        This is important in trying to fit a contour to a possibly-reflected form.
        """
        offsets = numpy.linspace(0, self.points.shape[0], align_steps, endpoint = False).astype(int)
        max_iters = int(numpy.ceil(0.1 * self.points.shape[0] / align_steps))
        best_distance = numpy.inf
        for offset in offsets:
            contour = self.as_offset_points(offset)
            if quick:
                distance = contour.procustes_distance_from(reference, True, weights, allow_reflection, allow_scaling, allow_translation)
            else:
                distance = contour.local_best_alignment(reference, weights, allow_reflection, allow_scaling, allow_translation, max_iters)
            if allow_reversed_orientation:
                rev = self.as_reversed_orientation()
                rev.offset_points(offset)
                if quick:
                    r_distance = rev.procustes_distance_from(reference, True, weights, allow_reflection, allow_scaling, allow_translation)
                else:
                    r_distance = rev.local_best_alignment(reference, weights, allow_reflection, allow_scaling, allow_translation, max_iters)
                if r_distance < distance:
                    contour = rev
                    distance = r_distance
            if distance < best_distance:
                best_offset = offset
                best_distance = distance
                best_contour = contour
        # copy best_contour to self
        self.__init__(other = best_contour)
        # now one last align step
        return self.local_best_alignment(reference, weights, allow_reflection, allow_scaling, allow_translation)

    def find_shape_intersections(self, ray_starts, ray_ends):
        """Find the closest points of intersection with the contour and a set of
        rays. Each ray must be represented as a start point and an end point.
        For each ray, two values are returned: the relative distance along the ray
        (as a fraction of the distance from start to end; could be negative) of
        the closest point, and the relative distance of the next-closest point
        that is on the ohter side of the contour. (That is, the point between the
        two returned intersection points is guaranteed to be INSIDE the contour.)
        If there are no intersection points, nans are returned.
        If the ray is exactly tangent to the contour, the results are undefined!
        No effort has been made to handle this uncommon case correctly.
        Also, the approximate contour position of the intersections is returned
        in a second array.
        """
        s0 = self.points
        s1 = numpy.roll(s0, -1, axis = 0)
        intersections = []
        point_numbers = []
        all_points = numpy.arange(len(self.points))
        for start, end in zip(ray_starts, ray_ends):
            radii, positions = utility_tools.line_intersections(start, end, s0, s1)
            intersects_in_segment = (positions <= 1) & (positions >= 0)
            intersect_radii = radii[intersects_in_segment]
            intersect_positions = (all_points + positions)[intersects_in_segment]
            if len(intersect_radii) == 0:
                intersections.append((None, None))
                point_numbers.append((None, None))
                continue
            pos_intersects = intersect_radii >= 0
            neg_intersects = ~(pos_intersects)
            pos_vals = intersect_radii[pos_intersects]
            neg_vals = intersect_radii[neg_intersects]
            pos_ord = numpy.argsort(pos_vals)
            neg_ord = numpy.argsort(neg_vals)
            pos = pos_vals[pos_ord]
            neg = neg_vals[neg_ord]
            pos_positions = intersect_positions[pos_intersects][pos_ord]
            neg_positions = intersect_positions[neg_intersects][neg_ord]
            closest = intersect_radii[numpy.absolute(intersect_radii).argmin()]
            if len(pos) % 2 == 1:
                # ray start is inside contour
                if closest >= 0:
                    closest_p = pos_positions[0]
                    next = neg[-1]
                    next_p = neg_positions[-1]
                else:
                    closest_p = neg_positions[-1]
                    next = pos[0]
                    next_p = pos_positions[0]
            else:
                # ray start is outside contour
                if closest >= 0:
                    closest_p = pos_positions[0]
                    next = pos[1]
                    next_p = pos_positions[1]
                else:
                    closest_p = neg_positions[-1]
                    next = neg[-2]
                    next_p = neg_positions[-2]
            intersections.append((closest, next))
            point_numbers.append((closest_p, next_p))
        return numpy.array(intersections, dtype=float), numpy.array(point_numbers, dtype=float)

    def find_nearest_point(self, point):
        """Find the position, in terms of the (fractional) contour parameter, of
        the point on the contour nearest to the given point."""
        s0 = self.points
        s1 = numpy.roll(s0, -1, axis = 0)
        closest_points, positions = utility_tools.closest_point_to_lines(point, s0, s1)
        positions.clip(0, 1)
        positions += numpy.arange(len(self.points))
        square_distances = ((point[:, numpy.newaxis] - closest_points)**2).sum(axis=1)
        return positions[square_distances.argmin()]

    def find_contour_midpoints(self, p1, p2):
        """Returns the two points midway between the given points along the
        contour, and the distances (in terms of the contour parameter)
        from the first point given to the two mid-points."""
        l = self.points.shape[0]
        if p2 < p1:
            p1, p2 = p2, p1
        ca = (p2 + p1)/2
        da = ca - p1
        return (ca, (ca - l/2.)%l), (da, da - l/2.)

    as_reversed_orientation = _copymethod(reverse_orientation)
    as_offset_points = _copymethod(offset_points)
    as_resampled = _copymethod(resample)
    as_globally_reordered_points = _copymethod(global_reorder_points)
    as_locally_reordered_points = _copymethod(local_reorder_points)
    as_locally_best_alignment = _copymethod(local_best_alignment)
    as_globally_best_alignment = _copymethod(global_best_alignment)

class ContourAndLandmarks(Contour):
    """Class for dealing with contour data that also has specific landmark points
    that should be taken account of when aligning with other contours."""
    _instance_data = dict(Contour._instance_data)
    _instance_data.update({'landmarks':numpy.zeros((0, 2)), 'weights':1})

    def _pack_landmarks_into_points(self):
        """Concatenate the list of landmarks to the list of points."""
        self.points = numpy.concatenate((self.points, self.landmarks))

    def _unpack_landmarks_from_points(self):
        """Unpack the list of landmarks from the list of points."""
        num_landmarks = len(self.landmarks)
        if num_landmarks == 0:
            return
        self.landmarks = self.points[-num_landmarks:]
        self.points = self.points[:-num_landmarks]

    def _get_points_and_landmarks(self):
        """Get the points and landmarks as a single concatenated list."""
        return numpy.concatenate((self.points, self.landmarks))

    def set_weights(self, landmark_weights):
        """Set the weights associted with the landmarks.

        In a landmark contour, each point and landmark can be associated with a
        weight, such that the total weight sums to one. This function is used to
        set the weights of the landmarks.

        If a single number (or list of size 1) is provided, then this weight is
        divided among all of the landmarks, and the remaining weight is divided
        among all of the contour points. Otherwise, the provided weights must be
        a list as long as the number of landmarks; each landmark will be given the
        corresponding weight, and the remaining weight will be divided among the
        contour points."""
        num_points = len(self.points)
        num_landmarks = len(self.landmarks)
        landmark_weights = numpy.asarray(landmark_weights, dtype=float)
        try:
            l = len(landmark_weights)
        except:
            l = 1
            landmark_weights = numpy.array([landmark_weights])
        if l == 1:
            landmark_weights /= num_landmarks
            landmark_weights = numpy.ones(num_landmarks) * landmark_weights
            l = num_landmarks
        elif l != num_landmarks:
            raise ValueError('Either one weight for all landmarks must be provided, or enough weights for each. (%d required, %d found)'%(num_landmarks, l))
        total_landmark_weight = landmark_weights.sum()
        if total_landmark_weight > 1:
            raise ValueError('The total weight assigned to the landmarks must not be greater than one.')
        point_weight = (1.0 - total_landmark_weight) / num_points
        point_weights = numpy.ones(num_points) * point_weight
        self.weights = numpy.concatenate((point_weights, landmark_weights))

    def transform(self, transform):
        self._pack_landmarks_into_points()
        Contour.transform(self, transform)
        self._unpack_landmarks_from_points()
    transform.__doc__ = Contour.transform.__doc__

    def rigid_align(self, reference, weights = None, allow_reflection = False, allow_scaling = False, allow_translation = True):
        if not isinstance(reference, ContourAndLandmarks):
            return Contour.rigid_align(self, reference, weights, allow_reflection, allow_scaling, allow_translation)
        if weights is None:
            weights = self.weights
        self._pack_landmarks_into_points()
        reference._pack_landmarks_into_points()
        Contour.rigid_align(self, reference, weights, allow_reflection, allow_scaling, allow_translation)
        self._unpack_landmarks_from_points()
        reference._unpack_landmarks_from_points()
    rigid_align.__doc__ = Contour.rigid_align.__doc__

    def rms_distance_from(self, reference):
        if not isinstance(reference, ContourAndLandmarks):
            return Contour.rms_distance_from(self, reference)
        return numpy.sqrt(((self.weights[:, numpy.newaxis] * (self._get_points_and_landmarks() - reference._get_points_and_landmarks()))**2).mean())
    rms_distance_from.__doc__ = Contour.rms_distance_from.__doc__

    def procustes_distance_from(self, reference, apply_transform = True,
            weights = None, allow_reflection = False, allow_scaling = False, allow_translation = True):
        if not isinstance(reference, ContourAndLandmarks):
            return Contour.procustes_distance_from(self, reference, apply_transform, weights,
                allow_reflection, allow_scaling, allow_translation)
        if weights is None:
            weights = self.weights
        self._pack_landmarks_into_points()
        reference._pack_landmarks_into_points()
        ret = Contour.procustes_distance_from(self, reference, apply_transform, weights,
            allow_reflection, allow_scaling, allow_translation)
        self._unpack_landmarks_from_points()
        reference._unpack_landmarks_from_points()
        return ret
    procustes_distance_from.__doc__ = Contour.procustes_distance_from.__doc__

    as_weighted = _copymethod(set_weights)

class PCAContour(Contour):
    """Class for storing the principal modes of shape variation from a set of
    contours.
    """
    _instance_data = dict(Contour._instance_data)
    _instance_data.update({'mean':numpy.zeros((0, 2)), 'modes':numpy.zeros((0, 0, 2)),
         'standard_deviations':numpy.zeros(0), 'total_variance':0, 'position':numpy.zeros(0)})

    def from_contours(cls, contours, required_variance_explained = 0.95, return_positions = False):
        """This class method should be used to construct a PCAContour object from a
        set of contours. The proncipal components of the contours are caltulated, and
        enough retained to account for the 'required_variance_explained' fraction.

        The mean shape is stored in the instance variable 'mean' and the principal components
        in 'modes'. The square root of the variance explained by each mode is stored in
        'standard_deviations', while the total variance of all of the modes (even those
        not retained) is in 'total_variance'.

        If the 'return_positions' parameter is true, then a tuple is returned containing
        (pca_contour, positions, normalized_positions), where 'pca_contour' is the new
        PCAContour instance, 'positions' is the position of each of the input contours
        along each principal shape mode, and 'normalized_positions' is the position
        along each mode in terms of standard deviations along that mode.
        """
        from celltool.numerics import pca
        data = [c.points for c in contours]
        if not utility_tools.all_same_shape(data):
            raise ValueError('All contours must have the same number of points in order to perform PCA.')
        units = [c.units for c in contours]
        if not numpy.all([u == units[0] for u in units]):
            raise ValueError('All contours must have the same units in order to produce a PCA shape model from them.')
        units = units[0]
        scales = [utility_tools.decompose_homogenous_transform(c.to_world_transform)[1] for c in contours]
        if numpy.all([numpy.allclose(scales[0], s) for s in scales[1:]]):
            transform = utility_tools.make_homogenous_transform(transform=scales[0])
        else:
            transform = numpy.eye(3)
        vals = pca.pca_dimensionality_reduce(numpy.array(data, dtype=numpy.float32), required_variance_explained)
        mean, pcs, norm_pcs, variances, total_variance, positions, norm_positions = vals
        c = cls(points=mean, mean=mean, modes=pcs, standard_deviations=numpy.sqrt(variances),
            total_variance=total_variance, position=numpy.zeros(len(norm_pcs)), units=units,
            to_world_transform=transform)
        if return_positions:
            return c, positions, norm_positions
        else:
            return c
    from_contours = classmethod(from_contours)

    def points_at_position(self, position, normalized = True):
        """Return the shape at a particular position along the principal shape modes.

        The 'position' parameter should be a list of numbers, one for each principal
        shape mode. The returned list of points (which could be used to construct
        a Contour object, if desired) is the shape at that position in PCA shape
        space. If the 'normalized' parameter is True, then the positions will be
        interpreted as standard deviations along each mode; otherwise they will be
        interpreted in the arbitrary shape-space units.
        """
        position = numpy.asarray(position)
        if normalized:
            position *= self.standard_deviations
        offsets = position[:, numpy.newaxis, numpy.newaxis] * self.modes
        return self.mean + offsets.sum(axis = 0)

    def set_position(self, position, normalized = True):
        """Set the 'points' instance variable to contain the shape at a specified
        position in PCA shape space. See the documentation for points_at_position
        for more details.
        """
        self.points = self.points_at_position(position, normalized)

    def find_position(self, contour, normalized = True):
        """Find the position of a contour object in terms of the PCA shape space
        represented by this PCAContour.

        If 'normalized' is True, then the position returned will be in terms of
        standard deviations along the shape axes; otherwise it will be in arbitrary
        shape-space units.
        """
        mean_offset = contour.points - self.mean
        position = (self.modes * mean_offset).sum(axis = -1).sum(axis = -1)
        if normalized:
            position /= self.standard_deviations
        return position

    def transform(self, transform):
        """Transform the data points with the provided affine transform.

        The transform should be a 3x3 transform in homogenous coordinates that will
        be used to transform row vectors by pre-multiplication.
        (E.g. final_point = transform * initial_point, where final_point and
        initial_point are 3x1, and * indicates matrix multiplication.)

        The provided transform is inverted and used to update the to_world_transform
        instance variable. This variable keeps track of all of the transforms performed
        so far.
        """
        Contour.transform(self, transform)
        self.mean = utility_tools.homogenous_transform_points(self.mean, transform)
        # don't translate the modes -- just scale/rotate them as required (translation being meaningless)
        scale_rotate = numpy.array(transform, copy=True)
        scale_rotate[2, :2] = 0
        self.modes = numpy.array([utility_tools.homogenous_transform_points(mode, scale_rotate) for mode in self.modes])

    def offset_points(self, offset):
        """Offset the point ordering forward or backward.

        Example: if the points are offset by 1, then the old points[0] is now at points[1],
        the old points[-1] is at points[0], and so forth. This doesn't change the spatial
        position of the contour, but it changes how the points are numbered.
        """
        self.points = numpy.roll(self.points, offset, axis=0)
        self.mean = numpy.roll(self.mean, offset, axis=0)
        self.modes = numpy.roll(self.modes, offset, axis=1)

    variances = property(lambda self:self.standard_deviations**2)

    as_position = _copymethod(set_position)
    as_offset_points = _copymethod(offset_points)

class CentralAxisContour(Contour):
    """This class stores contours that also have a central axis defined.

    Internally, the axis is stored in terms of a starting point, an ending
    point, and paired points along the contour between the start and end ( all
    defined in terms of their parametric position on the contour).
    These values are stored in the attribute "axis_positions", which is a 1D
    array containing the starting point, the points along one side from start
    to end, the ending point, and the points along the other side from end to
    start.

    In addition, the central_axis, top_points, and bottom_points attributes
    store the spatial positions of these points. Technically, these can be
    calculated from the axis_positions, but it's convenient to have those values
    pre-calculated.
    """
    _instance_data = dict(Contour._instance_data)
    _instance_data.update({'axis_positions': numpy.zeros((0,)), 'central_axis': numpy.zeros((0, 2)),
        'top_points': numpy.zeros((0, 2)), 'bottom_points': numpy.zeros((0, 2))})

    def from_contour(cls, contour, start, end, num_points, scale_steps=5, torsion_step=0.001,
                 spacing_step=0.001, curvature_step=0.04, overlap_step=0.1, endpoint_step=0.001, record=False):
        if num_points < 7:
            raise ValueError('num_points must be at least 7.')
        initial_start, initial_end = start, end
        try:
            num_steps = len(scale_steps)
        except:
            num_steps = scale_steps
            scale_steps = numpy.linspace(7, num_points, scale_steps, endpoint=True, dtype=int)
        from scipy.interpolate import fitpack
        tck, uout = contour.to_spline()
        start_pos, end_pos = numpy.transpose(fitpack.splev([start, end], tck))
        contour = cls(other=contour, central_axis=numpy.array([start_pos, end_pos]))
        num_subdivisions = int(numpy.log2(scale_steps[0] - 1))
        for i in range(num_subdivisions):
            contour.central_axis = contour.subdivide_axis()
        data = []
        for points in scale_steps:
            tck, uout = contour.axis_to_spline(spacing_corrected=True)
            new_axis = numpy.transpose(fitpack.splev(numpy.linspace(0, uout[-1], points, endpoint=True), tck))
            contour.estimate_axis_positions(new_axis, (start, end))
            data.append(contour.center_and_space_axis(max_iters=300, min_rms_change=0.001,
                torsion_step=torsion_step, spacing_step=spacing_step, curvature_step=curvature_step,
                overlap_step=overlap_step, endpoint_step=endpoint_step, record=record))
            start = contour.axis_positions[0]
            end = contour.axis_positions[points - 1]
        l = len(contour.points)
        if (min((start - initial_start)%l, l-(initial_start - start)%l) >
                min((end - initial_start)%l, l-(initial_start - end)%l)) :
            # if the initial "start" point is closer to the final end point than
            # the final start point, we should reverse the axis
            contour.reverse_central_axis()
        if record:
            return contour, data
        return contour
    from_contour = classmethod(from_contour)

    def recalculate_central_axis(self, tck=None):
        if tck is None:
            tck, uout = self.to_spline()
        from scipy.interpolate import fitpack
        n_pairs = (len(self.axis_positions)- 2) // 2
        spatial_pos = numpy.transpose(fitpack.splev(self.axis_positions, tck))
        start_p = spatial_pos[0:1]
        self.top_points = spatial_pos[1:n_pairs+1]
        end_p = spatial_pos[n_pairs+1:n_pairs+2]
        self.bottom_points = spatial_pos[n_pairs+2:][::-1]
        midpoints = (self.top_points + self.bottom_points) / 2
        self.central_axis = numpy.concatenate([start_p, midpoints, end_p])

    def reverse_central_axis(self):
        self.axis_positions = numpy.roll(self.axis_positions, len(self.axis_positions)//2)
        self.central_axis = self.central_axis[::-1]
        self.top_points, self.bottom_points = self.bottom_points[::-1], self.top_points[::-1]

    def estimate_axis_positions(self, axis, endpoints=None):
        axis_der = axis[2:] - axis[:-2]
        normals = numpy.empty(axis_der.shape)
        normals[:,0] = axis_der[:,1]
        normals[:,1] = -axis_der[:,0]
        if endpoints is not None:
            start, end = endpoints
        else:
            start = self.find_nearest_point(axis[0])
            end = self.find_nearest_point(axis[-1])
        intersect_points, arc_pos = self.find_shape_intersections(axis[1:-1], axis[1:-1]+normals)
        arc_pos = numpy.concatenate([[start, end], arc_pos.flatten()])
        arc_pos.sort()
        self.axis_positions = numpy.roll(arc_pos, -numpy.searchsorted(arc_pos, start))
        # if start < end:
        #     top = arc_pos[(arc_pos > start) & (arc_pos < end)]
        #     bottom = numpy.concatenate([arc_pos[arc_pos > end], arc_pos[arc_pos < start]])
        # else:
        #     top = arc_pos[(arc_pos > end) & (arc_pos < start)][::-1]
        #     bottom = numpy.concatenate([arc_pos[arc_pos > start], arc_pos[arc_pos < end]])[::-1]
        # self.axis_positions = numpy.concatenate([[start], top, [end], bottom])

    def subdivide_axis(self):
        d1 = (self.central_axis[1:] - self.central_axis[:-1])
        midpoints = self.central_axis[:-1] + d1 / 2.0
        normals = numpy.empty((len(d1), 2))
        normals[:,0] = d1[:,1]
        normals[:,1] = -d1[:,0]
        intersect_points, arc_pos = self.find_shape_intersections(midpoints, midpoints+normals)
        mid_intersection = intersect_points.sum(axis=1) / 2.0
        new_midpoints = midpoints + mid_intersection[:, numpy.newaxis] * normals
        new_axis = numpy.empty((len(self.central_axis)+len(new_midpoints), 2), dtype=float)
        new_axis[::2] = self.central_axis
        new_axis[1::2] = new_midpoints
        return new_axis

    def center_and_space_axis(self, max_iters=500, min_rms_change=1e-6, torsion_step=0.001,
             spacing_step=0.001, curvature_step=0.04, overlap_step=0.1, endpoint_step=0.001,
             record=False):
        tck, uout = self.to_spline()
        n_pairs = (len(self.axis_positions)- 2) // 2
        l = len(self.points)
        spatial_forces = numpy.zeros((len(self.axis_positions), 2), float)
        top_force = spatial_forces[1:n_pairs+1]
        # reverse bottom forces to match up in order with the top of each line pair
        bottom_force = spatial_forces[n_pairs+2:][::-1]
        torsion = numpy.empty((n_pairs, 2), float)
        normals = numpy.empty((n_pairs, 2), float)
        length_scale = max(*self.size())
        torsion_step *= length_scale**2
        spacing_scale = self.length() / l
        overlap_step *= spacing_scale
        iters = 0
        ms_change = numpy.inf
        min_ms_change = min_rms_change**2
        transpose, concatenate, sqrt = numpy.transpose, numpy.concatenate, numpy.sqrt
        newaxis, roll, exp, sign = numpy.newaxis, numpy.roll, numpy.exp, numpy.sign
        searchsorted = numpy.searchsorted
        from scipy.interpolate.fitpack import splev
        axis_positions = self.axis_positions
        if record:
            all_positions = [numpy.array(axis_positions, copy=True)]
        start_i, top_i = 0, 1
        end_i, bottom_i = n_pairs+1, n_pairs+2
        while (iters < max_iters and ms_change > min_ms_change):
            spatial_pos = transpose(splev(axis_positions, tck))
            start_p = spatial_pos[start_i:top_i]
            top_p = spatial_pos[top_i:end_i]
            end_p = spatial_pos[end_i:bottom_i]
            bottom_p = spatial_pos[bottom_i:][::-1]

            midpoints = (top_p + bottom_p) / 2
            axis = concatenate([start_p, midpoints, end_p])
            axis_der = axis[2:] - axis[:-2]
            axis_der /= sqrt((axis_der**2).sum(axis=1))[:, newaxis]
            normals[:,0] = axis_der[:,1]
            normals[:,1] = -axis_der[:,0]
            TmB = top_p - bottom_p
            torsion_denom = (normals * TmB).sum(axis=1)**3
            torsion[:,0] = 2*(axis_der * TmB).sum(axis=1) * TmB[:,1] / torsion_denom
            torsion[:,1] = 2*(-axis_der * TmB).sum(axis=1) * TmB[:,0] / torsion_denom
            torsion *= -torsion_step
            # top_force[:] = -torsion
            # bottom_force[:] = torsion
            top_force[1:-1] = -torsion[1:-1]
            bottom_force[1:-1] = torsion[1:-1]

            minus_previous = axis[1:] - axis[:-1]
            minus_previous_norms = sqrt((minus_previous**2).sum(axis=1))
            # mean_norm = minus_previous_norms.mean()
            # mean_frac = 2*(mean_norm/minus_previous_norms - 1)
            # spacing_vals = spacing_step * mean_frac[:, numpy.newaxis] * minus_previous
            spacing_vals = spacing_step * -2 * minus_previous
            spacing = (spacing_vals[1:] - spacing_vals[:-1])
            top_force -= spacing
            bottom_force -= spacing

            endpoint_vec = minus_previous[[1, -2]]
            TmBend = TmB[[0, -1]]
            TmBdotEnd = (TmBend * endpoint_vec).sum(axis=1)[:, newaxis]
            TmBendSq = (TmBend**2).sum(axis=1)[:, newaxis]
            endpointSq = (endpoint_vec**2).sum(axis=1)[:, newaxis]
            denom = (endpointSq * TmBendSq**2)
            endpoint_der = endpoint_step * 2 * (endpoint_vec * TmBdotEnd * TmBendSq - TmBend * TmBdotEnd**2) / denom
            top_force[[0, -1]] = -endpoint_der
            bottom_force[[0, -1]] = endpoint_der

            # endpoint_vec = minus_previous[[1, -2]]
            # vec_start = axis[[1, -2]]
            # endpoints = axis[[0, -1]]
            # w = endpoints - vec_start
            # c1 = (w*endpoint_vec).sum(axis=1)
            # c2 = (endpoint_vec**2).sum(axis=1)
            # b = c1 / c2
            # endpoint_proj = vec_start + b[:, numpy.newaxis]*endpoint_vec
            # proj_vec = (endpoint_proj - endpoints)
            # # proj_vec *= numpy.sign((proj_vec * TmB[[0, -1]]).sum(axis=1))[:, numpy.newaxis]
            # proj_vec = numpy.roll(proj_vec, 1, axis=1)
            # proj_vec[:,0] *= -1
            # proj_vec[1,:] *= -1
            # proj_vec *= endpoint_step
            # top_force[[0, -1]] += proj_vec
            # bottom_force[[0, -1]] -= proj_vec

            curvatures = curvature_step*2*(minus_previous[1:] - minus_previous[:-1])
            curvature_vals = (curvatures[:-2] - 2*curvatures[1:-1] + curvatures[2:])
            #top_force[0] -= -2*curvatures[0] + curvatures[1]
            top_force[1:-1] -= curvature_vals
            #top_force[-1] -= curvatures[-2] - 2*curvatures[-1]
            #bottom_force[0] -= -2*curvatures[0] + curvatures[1]
            bottom_force[1:-1] -= curvature_vals
            #bottom_force[-1] -= curvatures[-2] - 2*curvatures[-1]

            point_minus_prev = spatial_pos - roll(spatial_pos, 1, axis=0)
            distances = sqrt((point_minus_prev**2).sum(axis=1))[:, newaxis]
            overlap_derivatives = exp(-distances/spacing_scale) * point_minus_prev / distances
            overlap = overlap_step*(roll(overlap_derivatives, -1, axis=0) - overlap_derivatives)
            # top_force -= overlap[top_i:end_i]
            # bottom_force -= overlap[bottom_i:][::-1]
            top_force[1:-1] -= overlap[top_i+1:end_i-1]
            bottom_force[1:-1] -= overlap[bottom_i+1:-1][::-1]

            pos_d = transpose(splev(axis_positions, tck, 1))
            pos_d /= (pos_d**2).sum(axis=1)[:, newaxis]
            contour_forces = (spatial_forces * pos_d).sum(axis=1)
            old_positions = axis_positions.copy()
            axis_positions += contour_forces
            axis_positions %= l
            # find the point midway between the first pair of points along the axis
            (c1, c2), (d1, d2) = self.find_contour_midpoints(axis_positions[1], axis_positions[-1])
            if abs(d1) < abs(d2):
                start = axis_positions[start_i] = c1
            else:
                start = axis_positions[start_i] = c2
            (c1, c2), (d1, d2) = self.find_contour_midpoints(axis_positions[n_pairs], axis_positions[n_pairs+2])
            if abs(d1) < abs(d2):
                end = axis_positions[end_i] = c1
            else:
                end = axis_positions[end_i] = c2
            axis_positions.sort()
            # if start < end:
            #     top = axis_positions[(axis_positions > start) & (axis_positions < end)]
            #     bottom = concatenate([axis_positions[axis_positions > end], axis_positions[axis_positions < start]])
            # else:
            #     top = axis_positions[(axis_positions > end) & (axis_positions < start)][::-1]
            #     bottom = concatenate([axis_positions[axis_positions > start], axis_positions[axis_positions < end]])[::-1]
            # axis_positions = concatenate([[start], top, [end], bottom])
            axis_positions = roll(axis_positions, -searchsorted(axis_positions, start))
            ms_change = (((axis_positions - old_positions))**2).mean()
            iters += 1
            if record:
                all_positions.append(numpy.array(axis_positions, copy=True))
        self.axis_positions = axis_positions
        self.recalculate_central_axis(tck)
        if record:
            return iters, sqrt(ms_change), all_positions
        return iters, sqrt(ms_change)

    def transform(self, transform):
        Contour.transform(self, transform)
        self.recalculate_central_axis()

    def axis_cumulative_distances(self):
        return numpy.add.accumulate(numpy.concatenate([[0],self.axis_interpoint_distances()]))

    def axis_length(self, begin=None, end=None):
        """Return the length of the axis, optionally within the inclusive range specified."""
        return self.axis_interpoint_distances()[begin:end].sum()

    def axis_best_line(self):
        """Return best-fit line parameters (slope, intercept) for the central
        axis points."""
        from scipy import stats
        slope, intercept = stats.linregress(self.central_axis)[:2]
        return slope, intercept

    def axis_deviations(self):
        """Return the central axis represented as positions along the best fit line
        through the axis (the x-values) and signed distances away from that line
        (the y-values).

        Returns: start, end, points
           start, end: endpoints of the best-fit line through the central axis
           points: x-values are distances along this line, y-values are signed
               distances from the line to the central axis points
        """
        a, b = self.axis_best_line()
        x = self.central_axis[:,0]
        xs = x.min()
        xe = x.max()
        ys = a*xs + b
        ye = a*xe + b
        start = numpy.array([xs, ys])
        end = numpy.array([xe, ye])
        y_out = utility_tools.signed_distances_to_line(self.central_axis, start, end)
        closest_points, parameters = utility_tools.closest_points_to_line(self.central_axis, start, end)
        total_distance = (((start - end)**2).sum())**0.5
        x_out = parameters * total_distance
        if x_out[0] > x_out[-1]:
            x_out = x_out[::-1]
            y_out = y_out[::-1]
        return start, end, numpy.transpose([x_out, y_out])

    def axis_sinusoid_fit(self):
        """Fit a sinusoid function y = A*sin(2*pi/W * x + delta) to the
        deviations from the central axis (calculated by axis_deviations).

        A is the amplitude, W is the wavelength, and delta is the phase in
        radians.

        Returns: A, W, delta, x, y, y_fit, sse
            where x and y are the coordinates of the axis deivations, y_fit is
            the sinusoid fit, and sse is the sum of squared error between y and
            y_fit.
        """
        import scipy.optimize as opt
        def gen_sine(x, A, W, delta):
            A = abs(A)
            return A * numpy.sin((2*numpy.pi/W) * x + delta)

        def ssd(a, b):
            return ((a-b)**2).sum()

        two_pi = 2*numpy.pi
        start, end, positions = self.axis_deviations()
        x, y = positions.T
        A0 = numpy.ptp(y) / 2
        W0 = self.axis_wavelength(min_distance=A0/3)
        deltas = numpy.linspace(0, two_pi, 12, endpoint=False)
        delta0 = deltas[numpy.argmin([ssd(y, gen_sine(x, A0, W0, d)) for d in deltas])]
        params = (A0, W0, delta0)
        (A, W, delta), cv = opt.curve_fit(gen_sine, x, y, params)
        A = abs(A)
        delta %= two_pi
        if delta > numpy.pi:
            delta -= two_pi
        y_fit = gen_sine(x, A, W, delta)
        return A, W, delta, x, y, y_fit, ssd(y, y_fit)

    def axis_interpoint_distances(self):
        return utility_tools.norm(self.central_axis[1:] - self.central_axis[:-1], axis = 0)

    def axis_baseline_distances(self):
        """Return the distances from each point along the central axis to the
        baseline determined by the axis endpoints.
        Distances are signed: positive is on one side of the baseline, negative
        the other."""
        return utility_tools.signed_distances_to_line(self.central_axis,
            self.central_axis[0], self.central_axis[-1])

    def axis_rmsd(self):
        """Return the root-mean-square deviation of the axis points from the
        baseline determined by the axis endpoints.
        Note that the deviations are centered around zero before calculating the
        RMSD -- this makes it easier to compare RMSDs between contours with
        central axes that are all on one side of the baseline to contours with
        central axes that oscillate around the baseline."""
        distances = self.axis_baseline_distances()
        distances -= distances.mean()
        return numpy.sqrt((distances**2).mean())

    def axis_extrema(self, min_distance=0):
        """Return the indices of the points which represent extrema of the central
        axis, in terms of its deviation from the baseline determined by the endpoints.
        The endpoints themselves are always considered extrema; other extrema must be at
        least min_distance away from the baseline to be considered."""
        distances = self.axis_baseline_distances()
        maxima = utility_tools.local_maxima(distances, endpoints_allowed=False)
        minima = utility_tools.local_maxima(-distances, endpoints_allowed=False)
        indices = numpy.concatenate([maxima, minima])
        distances = numpy.absolute(distances[indices])
        indices = indices[distances > min_distance]
        return numpy.sort(numpy.concatenate([[0, len(self.central_axis)-1], indices]))

    def axis_wavelength(self, min_distance=0):
        """Return the approximate "wavelength" of the central axis, determined by
        the positions of the extrema of that axis (in terms of deviation from the
        baseline determined by the endpoints). Extrema less than min_distance from
        the baseline are ignored."""
        points, positions = utility_tools.closest_points_to_line(self.central_axis,
            self.central_axis[0], self.central_axis[-1])
        total_len = numpy.sqrt(((self.central_axis[0] - self.central_axis[-1])**2).sum())
        positions *= total_len
        extrema_positions = positions[self.axis_extrema(min_distance)]
        inter_extrema_distances = extrema_positions[1:] - extrema_positions[:-1]
        return 2 * inter_extrema_distances.mean()

    def axis_diameters(self, begin=None, end=None):
        """Return the diameter of the contour the points along the axis, within
        the optional inclusive range."""
        if begin is None:
            begin = 0
        if end is None:
            end = len(self.central_axis)
        else:
            end += 1
        distances = numpy.sqrt(((self.top_points - self.bottom_points)**2).sum(axis=1))
        return numpy.concatenate([[0], distances, [0]])[begin:end]

    def axis_spline_derivatives(self, begin=None, end=None, derivatives=1):
        """Calculate derivative or derivatives of the central axis using a spline
        fit,    over only the half-open range specified by 'begin' and 'end'."""
        from scipy.interpolate import fitpack
        if begin is None:
            begin = 0
        if end is None:
            end = len(self.central_axis)
        try:
            l = len(derivatives)
            unpack = False
        except:
            unpack = True
            derivatives = [derivatives]
        tck, uout = self.axis_to_spline()
        ret = [numpy.transpose(fitpack.splev(list(range(begin, end)), tck, der=d)) for d in derivatives]
        if unpack:
            ret = ret[0]
        return ret

    def axis_curvatures(self, begin = None, end = None):
        """Calculate the curvatures of the central axis (1/r of the osculating
        circle at each point), optionally over only the slice specified
        by 'begin' and 'end'."""
        d1, d2 = self.axis_spline_derivatives(begin, end, [1,2])
        x1 = d1[:,0]
        y1 = d1[:,1]
        x2 = d2[:,0]
        y2 = d2[:,1]
        return (x1*y2 - y1*x2) / (x1**2 + y1**2)**(3./2)

    def axis_normalized_curvature(self, begin = None, end = None):
        """Return the mean of the absolute values of the curvatures over the given
        range, times the path length along that range. For a line, this value is
        zero. For less axes shapes, the value is higher."""
        return numpy.absolute(self.axis_curvatures(begin, end)).mean() * self.axis_length(begin, end)

    def axis_normals(self):
        perpendiculars = numpy.empty(self.central_axis.shape, dtype=float)
        perpendiculars[[0,-1]] = utility_tools.find_perp(self.central_axis[[0, -2]], self.central_axis[[1, -1]])
        bisectors = utility_tools.find_bisector(self.central_axis[:-2], self.central_axis[1:-1], self.central_axis[2:])
        perps = utility_tools.find_perp(self.central_axis[:-2], self.central_axis[2:])
        dots = (bisectors * perps).sum(axis=1)
        perpendiculars[1:-1] = bisectors * numpy.sign(dots)[..., numpy.newaxis]
        return perpendiculars

    def axis_to_spline(self, smoothing = 0, spacing_corrected = False, end_weight = None):
        from scipy.interpolate import fitpack
        # the fitpack smoothing parameter is an upper-bound on the TOTAL squared deviation;
        # ours is a bound on the MEAN squared deviation. Fix the mismatch:
        l = len(self.central_axis)
        smoothing = smoothing * l
        if spacing_corrected:
            cumulative_distances = self.axis_cumulative_distances()
            u = l * cumulative_distances / cumulative_distances[-1]
        else:
            u = numpy.arange(l)
        if end_weight is not None:
            weights = numpy.ones(l)
            weights[0] = weights[-1] = end_weight
        else:
            weights = None
        tck, uout = fitpack.splprep(x = self.central_axis.transpose(), u = u, s = smoothing, w = weights)
        return tck, uout

    def interpolate_axis_points(self, positions):
        """Use spline interpolation to determine the spatial positions at the
        axis positions specified (fractional positions are thus acceptable)."""
        from scipy.interpolate import fitpack
        tck, uout = self.axis_to_spline()
        return numpy.transpose(fitpack.splev(positions, tck))

    def axis_to_bezier(self, match_curves_to_points = False):
        from scipy.interpolate import fitpack
        tck, u = self.axis_to_spline()
        if match_curves_to_points:
            to_insert = numpy.setdiff1d(u, numpy.unique(tck[0]))
            for i in to_insert:
                tck = fitpack.insert(i, tck, per = False)
        return utility_tools.b_spline_to_bezier_series(tck, per = False)

    def axis_top_bottom_to_spline(self):
        """Return two splines, mapping position along the axis (in the range
        [0, num_points-1], where num_points is the number of points along the axis,
        including the endpoints) to the contour parameter of the corresponding
        positions on the top and bottom of the contour. The spline outputs will
        give values of the contour parameter that are outside of the range of the
        contour; thus it is important to take the mod of these values by of the
        number of points along the contour before using them further.
        """
        from scipy.interpolate import fitpack
        axis_positions = self.axis_positions.copy()
        axis_points = len(axis_positions)//2 + 1
        contour_points = len(self.points)
        start = axis_positions[0]
        axis_positions[axis_positions < start] += contour_points
        top = axis_positions[:axis_points]
        bottom = numpy.concatenate([axis_positions[axis_points-1:], [start+contour_points]])[::-1]
        position_vals = numpy.arange(axis_points)
        top_tck = fitpack.splrep(position_vals, top)
        bottom_tck = fitpack.splrep(position_vals, bottom)
        return top_tck, bottom_tck

    def resample_axis(self, num_points):
        from scipy.interpolate import fitpack
        top_tck, bottom_tck = self.axis_top_bottom_to_spline()
        axis_points = len(self.axis_positions)//2 + 1
        contour_points = len(self.points)
        positions = numpy.linspace(0, axis_points-1, num_points, endpoint=True)
        top_p = fitpack.splev(positions, top_tck) % contour_points
        bottom_p = fitpack.splev(positions, bottom_tck) % contour_points
        self.axis_positions = numpy.concatenate([top_p, bottom_p[-2:0:-1]])
        self.recalculate_central_axis()

    def offset_points(self, offset):
        """Offset the point ordering forward or backward.

        Example: if the points are offset by 1, then the old points[0] is now at points[1],
        the old points[-1] is at points[0], and so forth. This doesn't change the spatial
        position of the contour, but it changes how the points are numbered.
        """
        Contour.offset_points(self, offset)
        self.axis_positions = (self.axis_positions + offset) % len(self.points)

    as_axis_centered_and_spaced = _copymethod(center_and_space_axis)
    as_axis_resampled = _copymethod(resample_axis)
    as_reversed_central_axis = _copymethod(reverse_central_axis)


def calculate_mean_contour(contours):
    """Calculate the average of a set of contours, while retaining units and
    scaling information, if possible. If all contours have associated landmarks,
    then the average will be such a contour as well."""
    all_points = [c.points for c in contours]
    if not utility_tools.all_same_shape(all_points):
        raise ContourError('Cannot calculate mean of contours with different numbers of points.')
    mean_points = numpy.mean(all_points, axis=0)
    units = [c.units for c in contours]
    if not numpy.all([u == units[0] for u in units]):
        raise ContourError('All contours must have the same units in order calculate their mean.')
    units = contours[0].units
    scales = [utility_tools.decompose_homogenous_transform(c.to_world_transform)[1] for c in contours]
    if numpy.all([numpy.allclose(scales[0], s) for s in scales[1:]]):
        transform = utility_tools.make_homogenous_transform(transform=scales[0])
    else:
        transform = numpy.eye(3)
    if numpy.all([isinstance(c, ContourAndLandmarks) for c in contours]):
        # if they're all landmark'd contours
        all_landmarks = [c.landmarks for c in contours]
        if not utility_tools.all_same_shape(all_landmarks):
            raise ContourError('Cannot calculate mean of contours with different numbers of landmarks.')
        mean_landmarks = numpy.mean(all_landmarks, axis=0)
        mean_weights = numpy.mean([c.weights for c in contours], axis=0)
        return ContourAndLandmarks(points=mean_points, units=units, landmarks=mean_landmarks,
            weights=mean_weights, to_world_transform=transform)
    else:
        return Contour(points=mean_points, units=units, to_world_transform=transform)


def from_file(filename, force_class=None):
    """Load a PointSet or subclass (e.g. Contour) from a file.

    This function can load objects previously saved with the to_file method. By
    default, the returned object will be of the type specified by the file; however
    if force_class is not None, then the object will be of that class. If it is
    not possible to force this (that is, if force_class is not a the same class or
    a superclass of the class specified in the file), then an error is raised."""
    data = {}
    original_class = None
    try:
        exec(compile(open(filename).read(), filename, 'exec'), numpy.__dict__, data)
        data = _compatibility_filter_data(data, force_class)
        module, class_name = data['cls']
        original_class = getattr(__import__(module, None, None, [class_name]), class_name)
        if force_class is not None:
            original_class = force_class
            if not issubclass(original_class, force_class):
                raise ContourError('Attepmting to load a saved file, originally of class "%s.%s", into incompatible class "%s.%s".'
                    % (module, class_name, force_class.__module__, force_class.__name__))
        c = original_class(other = data)
        c._filename = filename
        return c
    except Exception as e:
        if isinstance(e, KeyboardInterrupt):
            raise e
        if original_class is not None:
            raise IOError('Could not load file "%s" as a %s. (Error: %s)'%(filename, original_class.__name__, e))
        else:
            raise IOError('Could not load file "%s" as any kind of Contour. (Error: %s)'%(filename, e))

def _compatibility_filter_data(data, desired_class=None):
    if 'cls' not in data:
        # loading an old-version contour or PCA contour file
        if 'pcs' in data:
            return _filter_old_pca_contour(data)
        elif 'to_world_translation' in data:
            return _filter_old_contour(data)
        else:
            raise ContourError("Cannot determine type of old-style contour file!")
    else:
        return data

def _filter_old_contour(data):
    new_data = {}
    new_data['points'] = numpy.array(data['points'])
    new_data['to_world_transform'] = utility_tools.make_homogenous_transform(transform = data['to_world_transform'],
            translation = data['to_world_translation'] )
    new_data['units'] = ''
    new_data['cls'] = Contour.__module__, Contour.__name__
    return new_data

def _filter_old_pca_contour(data):
    new_data = {}
    new_data['points'] = new_data['mean'] = numpy.array(data['mean'])
    new_data['standard_deviations'] = numpy.sqrt(data['variances'])
    new_data['total_variance'] = data['total_variance']
    new_data['units'] = ''
    new_data['modes'] = numpy.array(data['pcs'])
    new_data['position'] = numpy.zeros(new_data['modes'].shape[0])
    new_data['cls'] = PCAContour.__module__, PCAContour.__name__
    return new_data