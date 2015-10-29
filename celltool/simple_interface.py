# Copyright 2007 Zachary Pincus
# This file is part of CellTool.
#
# CellTool is free software; you can redistribute it and/or modify
# it under the terms of version 2 of the GNU General Public License as
# published by the Free Software Foundation.

from celltool.utility.terminal_tools import progress_list, IndeterminantProgressBar
from celltool.contour import contour_class
from celltool.contour import contour_tools
from celltool.utility import warn_tools
from celltool.utility import path
import numpy


def load_contours(filenames, contour_type = None, show_progress = False):
    """Load a set of contour objects (instances of the classes defined in
    celltool.contour.contour_class) from disk.

    Parameters:
        - filenames: a list of file names of contour files. These files should
                have been previously saved to disk with the contours' to_file method, or
                with the 'save_contours' function from this module.
        - contour_type: attempt to force each contour loaded to be of this specified
                class. If this is not possible, an exception is raised.
        - show_progress: display a simple progress bar during this process.

    Returns a list of contours.
    """
    if show_progress:
        filenames = progress_list(filenames, 'Loading Contour Files')
    return [contour_class.from_file(name, force_class=contour_type) for name in filenames]

def save_contours(contours, filenames, show_progress = False):
    """Save a set of contour objects (instances of the classes defined in
    celltool.contour.contour_class) to disk.

    Parameters:
        - contours: a list of contour instances.
        - filenames: a list of file names for the contour files.
        - show_progress: display a simple progress bar during this process.
    """
    contours_and_names = list(zip(contours, filenames))
    if show_progress:
        contours_and_names = progress_list(contours_and_names, 'Saving Contour Files', lambda c_and_n: c_and_n[1])
    for c, n in contours_and_names:
        c.to_file(n)

def save_contour_data_for_matlab(contours, filenames, show_progress = False):
    from scipy import io
    contours_and_names = list(zip(contours, filenames))
    if show_progress:
        contours_and_names = progress_list(contours_and_names, 'Saving Matlab Files', lambda c_and_n: c_and_n[1])
    for c, n in contours_and_names:
        contour_attributes = list(c._instance_data.keys())
        contour_data = {}
        for attribute in contour_attributes:
            value = getattr(c, attribute)
            if isinstance(value, (numpy.ndarray, float, int)):
                contour_data[attribute] = value
        io.savemat(n, contour_data, appendmat=True, format='5')


def extract_contours(filenames, contour_value = None, min_area = None, max_area = None, show_progress = False):
    """Extract iso-value contours from a set of images.

    Parameters:
        - filenames: a list of names of image files.
        - contour_value: the pixel intensity value at which to extract the contours.
                If None, the mid-point intensity will be used.
        - min_area, max_area: area values (in pixels) above and below which contours
                will be discarded.
        - show_progress: display a simple progress bar during this process.

    Reurns, for each image, a list of the contours found in that image. (Thus a
    list of lists is returned.)
    """
    from celltool.utility import image
    if show_progress:
        filenames = progress_list(filenames, 'Extracting Contours from Images')
    axis_align = False
    closed_only = True
    all_contours = []
    for name in filenames:
        image_array = image.read_grayscale_array_from_image_file(name)
        contours = contour_tools.contours_from_image(image_array, contour_value, closed_only, min_area, max_area, axis_align)
        for contour in contours:
            contour._filename = name
        all_contours.append(contours)
    return all_contours

def resample_contours(contours, resample_points = 100, smoothing = 0, show_progress = False):
    """Resample a list of contours to have a specific number of evenly-spaced points.

    Parameters:
        - contours: a list of contour objects.
        - resample_points: the number of points each contour should have after resampling.
        - smoothing: the contour is smoothed before resampling; this value is the
                average distance from a smoothed point to the original contour point.
                Non-zero values allow pixel aliasing artifacts to be partially smoothed out.
        - show_progress: display a simple progress bar during this process.

    Reurns a list of the resampled contours.
    """
    if show_progress:
        contours = progress_list(contours, 'Resampling Contours', lambda c: c._filename)
    max_iters = 500
    min_rms_change = 1e-6
    step_size = 0.2
    return [contour.as_resampled(resample_points, smoothing, max_iters, min_rms_change, step_size) for contour in contours]

def find_centerlines(contours, centerline_points = 25, endpoints = 'horizontal', show_progress = False):
    """Finds the midlines of a set of contours and returns a new set of
    CentralAxisContour objects.

    The procedure is as follows: the initial start and end points of the axis
    are estimated according to the method specified in the "endpoints"
    parameter, and then a rough axis is fit between them. This axis is then
    numerically optimized to be evenly-spaced and centered at the contour
    midline, which also re-positions the end-points to more optimal locations.
    Then the axis is resampled to have the desired number of points, and is once
    more numerically optimized.

    Parameters:
        - contours: a list of contour objects.
        - centerline_points: the number of points each contour's central axis should have.
        - endpoints: controls how the initial end-points are selected:
                "distance" means that the two contour points that are most physically
                     distant will be selected.
                "vertical" means that the top-most/bottom-most points will be selected.
                "horizontal" means that the leftmost/rightmost points will be selected.
                "curvature" means that the points of maximal contour curvature will
                     be selected, under the constraint that the points must be separated
                     by at least 1/3 of the contour perimeter.
                (start, end): a pair of numbers indicates that these points are to be
                     used as the endpoints
        - show_progress: display a simple progress bar during this process.

    Reurns a list of the resampled contours.
    """
    if show_progress:
        contours = progress_list(contours, 'Finding Contour Centerlines', lambda c: c._filename)
    try:
        start, end = endpoints
        start, end = float(start), float(end)
        method = None
    except:
        if endpoints not in _centerline_methods:
            raise ValueError('"endpoints" parameter must be one of "distance", "vertical", "horizontal", "curvature", or a (start, end) pair')
        method = _centerline_methods[endpoints]
    centerline_contours = []
    for contour in contours:
        if method is not None:
            start, end = method(contour)
        new_contour = contour_class.CentralAxisContour.from_contour(contour, start, end, centerline_points)
        centerline_contours.append(new_contour)
    return centerline_contours

def _find_max_distance(contour):
    from celltool.numerics import utility_tools
    distances = utility_tools.squared_distance_matrix(contour.points)
    start, end = numpy.unravel_index(distances.argmax(), distances.shape)
    return start, end

def _find_vertical_bound(contour):
    start = contour.points[:,1].argmax()
    end = contour.points[:,1].argmin()
    return start, end

def _find_horizontal_bound(contour):
    start = contour.points[:,0].argmax()
    end = contour.points[:,0].argmin()
    return start, end

def _find_max_curvatures(contour):
    from celltool.numerics import utility_tools
    curvatures = contour.curvatures()
    if contour.signed_area() > 0:
        curvatures *= -1
    maxima = utility_tools.local_maxima(curvatures, cyclic=True)
    max_vals = curvatures[maxima]
    ordered_maxima = maxima[max_vals.argsort()][::-1]
    start = ordered_maxima[0]
    l = len(contour.points)
    min_dist = l / 3.
    for end in ordered_maxima[1:]:
        end_ok = (end - start) % l > min_dist and (start - end) % l > min_dist
        if end_ok:
            break
    if not end_ok:
        # no maxima in the allowed range... find the farthest one.
        distances = numpy.min([(ordered_maxima - start)%l, (start - ordered_maxima)%l], axis=1)
        end = ordered_maxima[distances.argmax()]
    return start, end

_centerline_methods = {
    'distance': _find_max_distance,
    'vertical': _find_vertical_bound,
    'horizontal': _find_horizontal_bound,
    'curvature': _find_max_curvatures
}


def transform_contours(contours, scale_factor = None, rotation = None, in_radians = True,
        units = None, new_zero_point = None, show_progress = False, title = 'Transforming Contours'):
    """Geometrically transform a list of contours.

    Parameters:
        - contours: a list of contour objects.
        - scale_factor: if not None, each contour will be scaled by this amount.
        - rotation: if not None, each contour will be rotated by this amount.
        - in_radians: if True, the rotation value is treated as radians, otherwise
                degrees.
        - units: the text name of the units that the contours should be assumed
                to be in (e.g. 'pixels' or 'mm').
        - new_zero_point: if not None, re-order the list of contour points so
                that this index becomes the first point in the list.
        - show_progress: display a simple progress bar during this process.
        - title: title of the progress display, if shown.

    Reurns a list of the transformed contours.
    """
    if show_progress:
        contours = progress_list(contours, title, lambda c: c._filename)
    ret = []
    for contour in contours:
        if scale_factor is not None: contour = contour.as_scaled(scale_factor)
        if rotation is not None: contour.rotate(rotation, in_radians)
        if units is not None: contour.units = units
        if new_zero_point is not None: contour.offset_points(-new_zero_point)
        ret.append(contour)
    return ret

def align_contours_to(contours, reference, global_align = True, align_steps = 8,
        allow_reflection = False, quick = False, show_progress = False):
    """Align a list of contours to a specific reference contour.

    Parameters:
        - contours: a list of contour objects.
        - reference: the reference contour to align to.
        - global_align: if True, the globally optimal point ordering and geometric
                alignment will be found to bring the contour into register with the
                reference. Otherwise only local hill-climbing will be used. Global
                alignment is slower than hill-climibing, however.
        - align_steps: if global_align is True, this is the number of different
                contour orientations to consider. For example, if align_steps = 8,
                then eight different (evenly-spaced) points will be chosen as the
                'first point' of the given contour, and then the fit to the reference
                will be locally optimized from that position. The best local fit is
                then treated as the global alignment.
        - allow_reflection: if True, then reflective transforms will be used if
                they make the alignment between the contour and reference better.
        - quick: if global_align is True and quick is True, then no local optimization
             will be performed at each of the global search steps. This will provide
             a rough and sub-optimal, but fast, alignment.
        - show_progress: display a simple progress bar during this process.

    Reurns a list of the aligned contours.

    For more specific control over the process, consider the function
    celltool.contour_tools.align_contour_to or the methods
    celltool.contour_class.Contour.global_best_alignment and local_best_alignment.
    """
    # copy contours so as not to change the originals.
    contours = contour_container = [c.as_copy() for c in contours]
    if show_progress:
        contours = progress_list(contours, 'Aligning Contours to Reference', lambda c: c._filename)
    allow_scaling = False
    weights = None
    for contour in contours:
        contour_tools.align_contour_to(contour, reference, global_align, align_steps, allow_reflection,
                allow_scaling, weights, quick)
    # return the original container, not the one that might have been turned into a progress_list...
    return contour_container

def align_contours(contours, align_steps = 8, allow_reflection = False, max_iters = 10, quick = False, show_progress = False):
    """Mutually align a set of contours to their mean in an expectation-maximization
    fashion.

    For each iteration, the mean contour is calculated, and then each contour is
    globally aligned to that mean with the celltool.contour_class.Contour.global_best_alignment
    method. Iteration continues until no contours are changed (beyond a given
    threshold), or the maximum number of iterations elapses.

    Parameters:
        - contours: a list of contour objects.
        - align_steps: The number of different contour orientations to consider
                when aligning each contour to the mean. For example, if align_steps = 8,
                then eight different (evenly-spaced) points will be chosen as the
                'first point' of the given contour, and then the fit to the mean
                will be locally optimized from that position. The best local fit is
                then treated as the global alignment.
        - allow_reflection: if True, then reflective transforms will be used if
                they make the alignment between the contour and reference better.
        - max_iters: maximum number of alignment iterations.
        - quick: if True, then no local optimization will be performed at each of
             the global search steps. This will provide a rough and sub-optimal, but
             fast, alignment.
        - show_progress: display a simple progress bar during this process.

    Reurns a list of the aligned contours.

    For more specific control over the process, consider the function
    celltool.contour_tools.align_contours.
    """
    # copy contours so as not to change the originals.
    contours = [c.as_copy() for c in contours]
    if show_progress:
        progress = IndeterminantProgressBar('Aligning Contours')
        l = len(contours)
        w = len(str(l))
        def callback(iters, i, changed):
            progress.update('Iteration %d (contour %*d / %*d) -- %*d contours changed.' %(iters+1, w, i+1, w, l, w, changed))
    else:
        callback = None
    allow_scaling = False
    weights = None
    min_rms_change = None
    iters = contour_tools.align_contours(contours, align_steps, allow_reflection, allow_scaling, weights, max_iters, min_rms_change, quick, callback)
    if iters == max_iters:
        warn_tools.warn('Contour alignment did not converge after %d iterations.'%max_iters)
    return contours

def grid_contours(contours, grid_shape=None):
    """Transform the contours so that each is centered in a distinct grid square.

    If specified, the (columns, rows) shape of the grid must be sufficient to
    contain the number of contours. Contours will be sorted by width."""
    import math
    contours = [c.as_copy() for c in contours]
    l = len(contours)
    if grid_shape is None:
        next_square = math.ceil(math.sqrt(l))
        cols, rows = next_square, next_square
    else:
        cols, rows = grid_shape
        if rows*cols < l:
            raise ValueError("The grid shape must be enough to contain each contour")
    sizes = numpy.array([contour.size() for contour in contours])
    sort = numpy.argsort(sizes, axis=0)
    max_sizes = sizes[sort[-1], (0, 1)]
    grid_size = max_sizes    + (0.1 * max_sizes.max())
    new_contours = []
    for contour_index, grid_index in zip(sort[:,0], numpy.ndindex(cols, rows)):
        grid_index = numpy.array(grid_index)
        contour = contours[contour_index]
        centroid = grid_index * grid_size + grid_size/2
        contour.recenter_bounds(centroid)
        new_contours.append(contour)
    return new_contours

def make_shape_model(contours, required_variance_explained = 0.95):
    """Make a PCA shape model from a set of contours.

    Parameters:
        - contours: a list of contour objects.
        - required_variance_explained: the fraction of total shape variance that
                should be explained by the returned PCA shape modes.

    Returns a (shape_model, header, rows, norm_header, norm_rows) tuple, where
    'shape_model' is an instance of the class celltool.contour_class.PCAContour,
    'header' is a "header row" for the 'rows' variable, which lists the name of
        each input contour and its position along each shape mode, and
    'norm_header' and 'norm_rows' are similar, but for the normalized positions,
        which are denominated in standard deviations from the mean shape along
        each mode.
    """
    return_positions = True
    shape_model, positions, norm_positions = contour_class.PCAContour.from_contours(contours, required_variance_explained, return_positions)
    norm_header = ['Contour'] + ['Mode %d (normalized)' %(i+1) for i in range(len(positions[0]))]
    header = ['Contour'] + ['Mode %d' %(i+1) for i in range(len(positions[0]))]
    norm_rows = []
    rows = []
    for c, p, n in zip(contours, positions, norm_positions):
        rows.append([c.simple_name()] + list(p))
        norm_rows.append([c.simple_name()] + list(n))
    return shape_model, header, rows, norm_header, norm_rows

def reorient_images(contours, image_names, new_names, pad_factor = 1.2, mask = True, show_progress = False):
    """Reorient a set of images to be aligned to the input contours.

    The image regions corresponding to each contour are clipped out of the images,
    reoriented, and saved to new files. Each output image is the same size.

    Parameters:
        - contours: a list of contour objects.
        - image_names: a list of images to load from disk.
        - new_names: the names of the output image files.
        - pad_factor: the size of the output images is calculated by taking the
                maximum size (in pixels) over all of the contours, and then multiplying
                by this pad_factor.
        - mask: if True, image pixels outside the contour will be zeroed out.
        - show_progress: display a simple progress bar during this process.
    """
    from celltool.utility import image
    _caching_reader.prime(image_names)
    # Put the contours into pixel units so that when we transform the images,
    # they aren't resized up or down to whatever the contour units are...
    contours = [c.as_descaled() for c in contours]
    contours_and_images = list(zip(contours, image_names, new_names))
    if show_progress:
        contours_and_images = progress_list(contours_and_images, 'Reorienting Images', lambda c_n_nn: c_n_nn[1])
    bboxes = [c.bounding_box() for c in contours]
    low = numpy.min([b[0] for b in bboxes], axis = 0)
    high = numpy.max([b[1] for b in bboxes], axis = 0)
    size = numpy.ceil((high - low) * pad_factor).astype(int)
    for c, n, nn in contours_and_images:
        image_array = _caching_reader.load(n)
        new_image = contour_tools.transform_image_to_contour(c, image_array, size, mask)
        image.write_array_as_image_file(new_image, nn)

def make_masks(contours, new_names, pad_factor = 1.2, inside = 255, outside = 0, show_progress = False):
    """Create binary masks from a set of contours.

    Parameters:
        - contours: a list of contour objects.
        - new_names: the names of the output image files.
        - pad_factor: the size of the output images is calculated by taking the
                maximum size (in pixels) over all of the contours, and then multiplying
                by this pad_factor.
        - inside, outside: pixel intensity values for the regions inside and outside
                of the contours.
        - show_progress: display a simple progress bar during this process.
    """
    from celltool.utility import image
    # Put the contours into pixel units
    contours = [c.as_descaled() for c in contours]
    contours_and_names = list(zip(contours, new_names))
    if show_progress:
        contours_and_names = progress_list(contours_and_names, 'Making Image Masks', lambda c_n: c_n[0]._filename)
    bboxes = [c.bounding_box() for c in contours]
    low = numpy.min([b[0] for b in bboxes], axis = 0)
    high = numpy.max([b[1] for b in bboxes], axis = 0)
    size = numpy.ceil((high - low) * pad_factor).astype(int)
    for c, nn in contours_and_names:
        mask = contour_tools.get_binary_mask(c, size)
        mask = numpy.where(mask, inside, outside).astype(numpy.uint8)
        image.write_array_as_image_file(mask, nn)

def add_image_landmarks_to_contours(contours, image_names, landmark_ranges, landmark_weights, image_type, show_progress = False):
    """Add landmark points from a set of images to a list of contours.

    Parameters:
        - contours: a list of contour objects.
        - image_names: a list of images to load from disk.
        - landmark_ranges: a list of (low, high) tuples, where the position of a
                given landmark is taken as the geometric centroid of all of the pixels
                within the contour falling within the intensity range of [low, high]
                (inclusive). This makes it easy for a user to 'paint' landmarks onto
                an image with particular intensities and then extract the centers of
                those landmarks (If landmarks outside of the contour are required, you
                will need to use the function celltool.contour_tools.add_image_landmarks,
                which this uses internally.)
        - landmark_weights: In a landmark contour, each point and landmark can be
                associated with a weight, such that the total weight sums to one.
                If a single number (or list of size 1) is provided, then this weight
                is divided among all of the landmarks, and the remaining weight is
                divided among all of the contour points. Otherwise, the provided
                weights must be a list as long as the number of landmark ranges; each
                landmark will be given the corresponding weight, and the remaining
                weight will be divided among the contour points.
        - image_type: If the image type is 'original' then the images are assumed
                to be congruent with those from which the contours were originally
                extracted. If the type is 'aligned' then the images are assumed to have
                been generated with the reorient_images function from this module, or
                celltool.contour.contour_tools.transform_image_to_contour, which that
                function calls.
        - show_progress: display a simple progress bar during this process.

    Reurns a list of new celltool.contour.contour_class.ContourAndLandmarks objects.
    """
    if image_type not in ('original', 'aligned'):
        raise RuntimeError("Image type %s is invalid. Must be 'original' or 'aligned'."%image_type)
    _caching_reader.prime(image_names)
    contours_and_images = list(zip(contours, image_names))
    if show_progress:
        contours_and_images = progress_list(contours_and_images, 'Adding Landmarks from Image', lambda c_n: c_n[0]._filename)
    landmark_contours = []
    for contour, image_name in contours_and_images:
        image_array = _caching_reader.load(image_name)
        lcontour = contour_tools.add_image_landmarks(contour, image_array, landmark_ranges, image_type, mask = True)
        lcontour.set_weights(landmark_weights)
        landmark_contours.append(lcontour)
    return landmark_contours

def reweight_landmarks(contours, weights, show_progress = False):
    """Re-weight the landmark points from a list of landmarked contours.

    Parameters:
        - contours: a list of ContourAndLandmarks objects.
        - weights: In a landmark contour, each point and landmark can be
                associated with a weight, such that the total weight sums to one.
                If a single number (or list of size 1) is provided, then this weight
                is divided among all of the landmarks, and the remaining weight is
                divided among all of the contour points. Otherwise, the provided
                weights must be a list as long as the number of landmark ranges; each
                landmark will be given the corresponding weight, and the remaining
                weight will be divided among the contour points.
        - show_progress: display a simple progress bar during this process.

    Reurns a list of new celltool.contour.contour_class.ContourAndLandmarks objects.
    """
    if show_progress:
        contours = progress_list(contours, "Re-Weighting Landmarks", lambda c: c._filename)
    ret = []
    for contour in contours:
        try:
            contour = contour.as_weighted(weights)
        except AttributeError:
            warn_tools.warn('Contour "%s" does not have any landmarks to re-weight.'%contour._filename)
        ret.append(contour)
    return ret


def measure_contours(contours, show_progress = False, *measurements):
    """Apply a set of measurements to a list of contours.

    Parameters:
        - contours: a list of contour objects.
        - show_progress: display a simple progress bar during this process.
    The remaining parameters are assumed to be "measurement" objects. These objects
        must have two methods: 'header', which produces a list of the names of the
        measurements that will be made (called as 'measurement.header(contours)'),
        and 'measure', which must produce a list of measurements (the same length
        as the header list) when called on a single contour object.

    Reurns a (header, measurements) tuple, where header is a list of the names
    of all the measurements made, and measurements is a list of the measurements
    made for each contour (a list of lists).

    Measurement objects defined in this module include Area, AspectRatio, Angle,
    Centroid, PathLength, NormalizedCurvature, ShapeModeMeasurement,
    SwathMeasurement, and ImageIntegration.
    """
    header = ['Contour']
    for measurement in measurements:
        header.extend(measurement.header(contours))
    if show_progress:
        contours = progress_list(contours, 'Measuring Contours', lambda c: c._filename)
    all_measurements = []
    for contour in contours:
        contour_measurements = [contour.simple_name()]
        for measurement in measurements:
            measure = measurement.measure(contour)
            contour_measurements.extend(measure)
        all_measurements.append(contour_measurements)
    return header, all_measurements

class _ContourMeasurement(object):
    @classmethod
    def header(cls, contours):
        return [cls._header]
    @classmethod
    def measure(cls, contour):
        return [cls._method(contour)]

class Area(_ContourMeasurement):
    """Measure the areas of contours."""
    _header = 'Area'
    _method = contour_class.Contour.area

class AspectRatio(_ContourMeasurement):
    """Measure the x-size/y-size aspect ratio of contours."""
    _header = 'Aspect Ratio'
    _method = contour_class.Contour.aspect_ratio

class AlignmentAngle(_ContourMeasurement):
    """Measure the current alignment angle of contours."""
    _header = 'Alignment Angle'
    _method = contour_class.Contour.alignment_angle

class Centroid(object):
    """Measure the x, y centroid of contours"""
    @staticmethod
    def header(contours):
        return ['x-centroid', 'y-centroid']
    @staticmethod
    def measure(contour):
        return list(contour.as_world().centroid())

class Size(object):
    """Measure the x, y size of contours"""
    @staticmethod
    def header(contours):
        return ['x-size', 'y-size']
    @staticmethod
    def measure(contour):
        return list(contour.size())

class _CentralAxisMeasurement(_ContourMeasurement):
    @classmethod
    def header(cls, contours):
        for contour in contours:
            if not isinstance(contour, contour_class.CentralAxisContour):
                raise TypeError("A contour with a central axis defined is required for this measurement")
        return [cls._header]

class AxisRMSD(_CentralAxisMeasurement):
    """Measure the deviation of the axis points from the baseline determined by
    the axis endpoints. Note that the deviations are re-centered around zero
    before the RMSD is calculated, so that contours with deviations only on one
    side of the baseline can be better compared with contours with deviations to
    both sides."""
    _header = 'Axis RMSD'
    _method = contour_class.CentralAxisContour.axis_rmsd

class RelativeAxisRMSD(_CentralAxisMeasurement):
    """Measure the deviation of the axis points from the baseline determined by
    the axis endpoints, relative to the length of that baseline. Note that the
    deviations are re-centered around zero before the RMSD is calculated, so
    that contours with deviations only on one side of the baseline can be better
    compared with contours with deviations to both sides."""
    _header = 'Relative Axis RMSD'
    @staticmethod
    def _method(contour):
        rmsd = contour.axis_rmsd()
        baseline_length = numpy.sqrt(((contour.central_axis[0]-contour.central_axis[-1])**2).sum())
        return rmsd / baseline_length

class AxisPeakAmplitude(_CentralAxisMeasurement):
    """Measure the maximal deviation of the axis points from the baseline
    determined by the axis endpoints. Note that the deviations are re-centered
    around zero before the peak is calculated, so that contours with deviations
    only on one side of the baseline can be better compared with contours with
    deviations to both sides."""
    _header = 'Peak Axis Amplitude'
    @staticmethod
    def _method(self, contour):
        distances = contour.axis_baseline_distances()
        distances -= distances.mean()
        return numpy.absolute(distances).max()

class AxisLengthRatio(_CentralAxisMeasurement):
    """Measure the ratio of the length of the axis to that of the axis baseline
    determined by its endpoints."""
    _header = 'Axis Length Ratio'
    @staticmethod
    def _method(contour):
        length = contour.axis_length()
        baseline_length = numpy.sqrt(((contour.central_axis[0]-contour.central_axis[-1])**2).sum())
        return length / baseline_length

class AxisWavelength(_CentralAxisMeasurement):
    """Measure the average "wavelength" of the central axis."""
    _header = 'Axis Wavelength'
    @staticmethod
    def _method(contour):
        max_width = contour.axis_diameters().max()
        return contour.axis_wavelength(min_distance=max_width/10.)

class AxisWavenumber(_CentralAxisMeasurement):
    """Measure the number of full oscillations of the central axis. Will be
    either an integer or an integral multiple of 0.5."""
    _header = 'Axis Wavenumber'
    @staticmethod
    def _method(contour):
        max_width = contour.axis_diameters().max()
        return len(contour.axis_extrema(min_distance=max_width/10.)) / 2.0 - 1

class _BoundedContourMeasurement(object):
    def __init__(self, begin=None, end=None):
        """Note: begin and end are zero-indexed, but the header information is generated
        for one-indexed points. This makes it easier to explain to non-cs people."""
        self.begin = begin
        self.end = end
    def header(self, contours):
        if self.begin is None and self.end is None:
            return [self._header]
        elif self.begin is None:
            return [self._header + ' ' + '(to %d)'%(self.end+1)]
        elif self.end is None:
            return [self._header + ' ' + '(from %d)'%(self.begin+1)]
        else:
            return [self._header + ' ' + '(from %d to %d)'%(self.begin+1, self.end+1)]
    def measure(self, contour):
        # look up _method on type(self) to retrieve unbound method, not method bound
        return [type(self)._method(contour, begin=self.begin, end=self.end)]

class PathLength(_BoundedContourMeasurement):
    """Measure the length of the contour from 'begin' to 'end'. These
         values can be 'None' to use the contour endpoints, and 'end' can be
         smaller than 'begin', which just gets a slice of the contour that
         wraps around the zero point.
    """
    _header = 'Path Length'
    _method = contour_class.Contour.length

class NormalizedCurvature(_BoundedContourMeasurement):
    """Measure the smoothness of the contour from 'begin' to 'end'. These
         values can be 'None' to use the contour endpoints, and 'end' can be
         smaller than 'begin', which just gets a slice of the contour that
         wraps around the zero point.
    """
    _header = 'Normalized Curvature'
    _method = contour_class.Contour.normalized_curvature

class SideCurvature(object):
    def __init__(self, begin_1, end_1, begin_2, end_2):
        self.begin_1 = begin_1
        self.end_1 = end_1
        self.begin_2 = begin_2
        self.end_2 = end_2
    @staticmethod
    def header(contours):
        return ['Side Curvature']
    def measure(self, contour):
        return [contour.normalized_curvature(self.begin_1, self.end_1)+contour.normalized_curvature(self.begin_2, self.end_2)]

class _BoundedCentralAxisMeasurement(_BoundedContourMeasurement):
    def header(self, contours):
        for contour in contours:
            if not isinstance(contour, contour_class.CentralAxisContour):
                raise TypeError("A contour with a central axis defined is required for this measurement")
        return _BoundedContourMeasurement.header(self, contours)

class AxisLength(_BoundedCentralAxisMeasurement):
    _header = 'Axis Length'
    _method = contour_class.CentralAxisContour.axis_length

class AxisMeanDiameter(_BoundedCentralAxisMeasurement):
    """Measure the mean diameter of the cell along the axis points specified (endpoints excluded by default)"""
    _header = 'Mean Axis Diameter'
    @staticmethod
    def _method(contour, begin, end):
        if begin is None:
            begin = 1
        if end is None:
            end = len(contour.central_axis)-1
        return contour.axis_diameters(begin, end).mean()

class AxisNormalizedCurvature(_BoundedCentralAxisMeasurement):
    _header = 'Normalized Axis Curvature'
    _method = contour_class.CentralAxisContour.axis_normalized_curvature

class AxisDiameters(object):
    """Measure the diameters of the cell along the axis points specified"""
    def __init__(self, begin=None, end=None):
        self.begin = begin
        self.end = end

    def header(self, contours):
        from celltool.numerics import utility_tools
        for contour in contours:
            if not isinstance(contour, contour_class.CentralAxisContour):
                raise TypeError("A contour with a central axis defined is required for this measurement")
        if not utility_tools.all_same_shape([c.central_axis for c in contours]):
            raise RuntimeError('All contours must have the same number of points along their central axes to make diameter measurements.')
        point_range = list(range(len(contours[0].central_axis)))
        if self.end:
            point_range = point_range[:self.end+1]
        if self.begin:
            point_range = point_range[self.begin:]
        return ['Diameter (point %d)' %(p+1) for p in point_range]

    def measure(self, contour):
        return contour.axis_diameters(self.begin, self.end)


class ShapeModeMeasurement(object):
    """Measure the positions of contours in the coordinates of a particular
    PCA shape model stored on disk as a PCAContour."""
    def __init__(self, shape_model_file, modes, normalized):
        """Parameters:
             - shape_model_file: file name of an on-disk PCAContour.
             - modes: a list of the modes to consider, or None to use all.
             - normalized: if True, return the positions in terms of standard deviations
                     from the mean shape.
        """
        self.shape_model = contour_class.from_file(shape_model_file, contour_class.PCAContour)
        if not modes:
            modes = list(range(1, len(self.shape_model.modes) + 1))
        self.modes = modes
        self.normalized = normalized
        for m in modes:
            if m > len(self.shape_model.modes) or m < 1:
                raise RuntimeError('Shape model %s does not have a mode %d.'%(shape_model_file, m))

    def header(self, contours):
        if self.normalized:
            norm = 'normalized '
        else:
            norm = ''
        return ['%s (%smode %d)' %(self.shape_model.simple_name(), norm, m) for m in self.modes]

    def measure(self, contour):
        positions = self.shape_model.find_position(contour, self.normalized)
        return positions.take([m-1 for m in self.modes])

class _ImageMeasurementBase(object):
    def __init__(self, image_type, contour_match, image_names):
        self.image_type = image_type
        self.contour_match = contour_match
        if self.contour_match == 'name':
            self.image_names = dict([(path.path(name).namebase, name) for name in image_names])
        else:
            self.image_names = image_names
        _caching_reader.prime(image_names)

    def _get_image_for_contour(self, contour):
        if self.contour_match == 'order':
            return self.image_names.pop(0)
        name = contour.simple_name()
        no_trailing_numbers = name.rsplit('-', 1)[0]
        if name in self.image_names:
            return self.image_names[name]
        elif no_trailing_numbers in self.image_names:
            return self.image_names[no_trailing_numbers]
        else:
            raise RuntimeError('Could not find matching image for contour "%s".'%contour._filename)


class SwathMeasurement(_ImageMeasurementBase):
    """Measure image intensities in regions defined by contours."""
    def __init__(self, measurement_name, begin, end, offset, depth, mode, samples, image_type, contour_match, image_names):
        """Parameters:
             - measurement_name: a text name for this measurement, reported in the header.
             - begin, end: contour points to define the length of the swath. These
                 values can be 'None' to use the contour endpoints, and 'end' can be
                 smaller than 'begin', which just gets a slice of the contour that
                 wraps around the zero point.
                 Note: begin and end are zero-indexed, but the header information is
                 generated for one-indexed points. This makes it easier to explain to
                 non-cs people.
             - offset: distance inside (or outside if this value is negative) of the
                 contour that the region should begin.
             - depth: depth distance of the image swath.
             - mode: if 'depth_profile', then the swath is averaged along the length
                     of the contour (from 'begin' to 'end') and a 1-d array of length
                     'samples' is returned. If 'length_profile', then the swath is averaged
                     along the depth dimension, and a 1-d array of length 'end'-'begin' is
                     returned. If 'grand_average' the the mean pixel intensity across the
                     entire swath is measured.
             - samples: the number of sample points in the 'depth' direction.
             - image_type: If the image type is 'original' then the images are assumed
                     to be congruent with those from which the contours were originally
                     extracted. If the type is 'aligned' then the images are assumed to have
                     been generated with the reorient_images function from this module, or
                     celltool.contour.contour_tools.transform_image_to_contour, which that
                     function calls.
             - contour_match: if 'name' then contour files are matched to the image
                     names by looking for matching filenames (sans directories and extensions).
                     If 'order', then the contours are matched to the image names by their
                     order in the list.
             - image_names: a list of filenames of the images associated with the contours.
        """
        if mode not in ('depth_profile', 'length_profile', 'grand_average'):
            raise RuntimeError("Mode %s is invalid. Must be one of 'depth_profile', 'length_profile', or 'grand_average'."%mode)
        if image_type not in ('original', 'aligned'):
            raise RuntimeError("Image type %s is invalid. Must be 'original' or 'aligned'."%image_type)
        if contour_match not in ('name', 'order'):
            raise RuntimeError("Contour match method %s is invalid. Must be one of 'name' or 'order'."%contour_match)

        self.measurement_name = measurement_name
        self.begin = begin
        self.end = end
        self.offset = offset
        self.depth = depth
        self.mode = mode
        self.samples = samples
        _ImageMeasurementBase.__init__(self, image_type, contour_match, image_names)

    def header(self, contours):
        from celltool.numerics import utility_tools
        if self.samples is None:
            samples = self.depth
        else:
            samples = self.samples
        if self.mode == 'depth_profile':
            return ['%s (depth %g)' %(self.measurement_name, d) for d in numpy.linspace(self.offset, self.offset + self.depth, samples, endpoint = True)]
        elif self.mode == 'length_profile':
            if not utility_tools.all_same_shape([c.points for c in contours]):
                raise RuntimeError('All contours must have the same number of points to make swath measurements.')
            point_range = utility_tools.inclusive_periodic_slice(list(range(len(contours[0].points))), self.begin, self.end)
            return ['%s (point %d)' %(self.measurement_name, p+1) for p in point_range]
        else:
            return ['%s (mean)' %self.measurement_name]

    def measure(self, contour):
        image_name = self._get_image_for_contour(contour)
        image_array = _caching_reader.load(image_name)
        swath = contour_tools.get_image_swath(contour, image_array, self.begin, self.end, self.offset,
            self.depth, d_samples=self.samples, image_type=self.image_type)
        if self.mode == 'depth_profile':
            ret = swath.mean(axis = 0).filled(numpy.nan)
        elif self.mode == 'length_profile':
            ret = swath.mean(axis = 1).filled(numpy.nan)
        else:
            ret = [swath.mean(axis = None)]
        if numpy.sometrue(numpy.isnan(ret)):
            warn_tools.warn("One portion of the swath for contour %s on image %s was entirely outside of the image region; these values will be reported as 'nan'."%(contour.simple_name(), image_name))
        return list(ret)

class AxisSwathMeasurement(_ImageMeasurementBase):
    """Measure image intensities in regions defined by contours."""
    def __init__(self, measurement_name, mode, begin, end, samples, image_type, contour_match, image_names):
        """Parameters:
             - measurement_name: a text name for this measurement, reported in the header.
             - begin, end: zero-indexed, inclusive points along the central axis (or None).
             - mode: if 'depth_profile', then the swath is averaged along the length
                     of the contour's central axis and a 1-d array of length 'samples'
                     is returned. If 'length_profile', then the swath is averaged along
                     the depth dimension, and a 1-d array of length 'end'-'begin' is
                     returned. If 'grand_average' the the mean pixel intensity across
                     the entire swath is measured.
             - samples: the number of sample points in the 'depth' direction.
             - image_type: If the image type is 'original' then the images are assumed
                     to be congruent with those from which the contours were originally
                     extracted. If the type is 'aligned' then the images are assumed to have
                     been generated with the reorient_images function from this module, or
                     celltool.contour.contour_tools.transform_image_to_contour, which that
                     function calls.
             - contour_match: if 'name' then contour files are matched to the image
                     names by looking for matching filenames (sans directories and extensions).
                     If 'order', then the contours are matched to the image names by their
                     order in the list.
             - image_names: a list of filenames of the images associated with the contours.
        """
        if mode not in ('depth_profile', 'length_profile', 'grand_average'):
            raise RuntimeError("Mode %s is invalid. Must be one of 'depth_profile', 'length_profile', or 'grand_average'."%mode)
        if image_type not in ('original', 'aligned'):
            raise RuntimeError("Image type %s is invalid. Must be 'original' or 'aligned'."%image_type)
        if contour_match not in ('name', 'order'):
            raise RuntimeError("Contour match method %s is invalid. Must be one of 'name' or 'order'."%contour_match)

        self.measurement_name = measurement_name
        self.begin = begin
        self.end = end
        self.mode = mode
        self.samples = samples
        _ImageMeasurementBase.__init__(self, image_type, contour_match, image_names)

    def header(self, contours):
        from celltool.numerics import utility_tools
        for contour in contours:
            if not isinstance(contour, contour_class.CentralAxisContour):
                raise TypeError("A contour with a central axis defined is required for this measurement")
        if self.mode == 'depth_profile':
            return ['%s (top-to-bottom fraction %g)' %(self.measurement_name, d) for d in numpy.linspace(0, 1, self.samples)]
        elif self.mode == 'length_profile':
            if not utility_tools.all_same_shape([c.central_axis for c in contours]):
                raise RuntimeError('All contours must have the same number of points along their central axes to make swath measurements.')
            point_range = list(range(len(contours[0].central_axis)))
            if self.end:
                point_range = point_range[:self.end+1]
            if self.begin:
                point_range = point_range[self.begin:]
            return ['%s (point %d)' %(self.measurement_name, p+1) for p in point_range]
        else:
            return ['%s (mean)' %self.measurement_name]

    def measure(self, contour):
        image_name = self._get_image_for_contour(contour)
        image_array = _caching_reader.load(image_name)
        swath = contour_tools.get_axis_swath(contour, image_array, self.samples, self.begin, self.end, image_type=self.image_type)
        if self.mode == 'depth_profile':
            ret = swath.mean(axis = 0).filled(numpy.nan)
        elif self.mode == 'length_profile':
            ret = swath.mean(axis = 1).filled(numpy.nan)
        else:
            ret = [swath.mean(axis = None)]
        if numpy.sometrue(numpy.isnan(ret)):
            warn_tools.warn("One portion of the swath for contour %s on image %s was entirely outside of the image region; these values will be reported as 'nan'."%(contour.simple_name(), image_name))
        return list(ret)

class ImageIntegration(_ImageMeasurementBase):
    """Measure image intensity within contours."""
    def __init__(self, measurement_name, image_type, contour_match, image_names):
        """Parameters:
             - measurement_name: a text name for this measurement, reported in the header.
             - image_type: If the image type is 'original' then the images are assumed
                     to be congruent with those from which the contours were originally
                     extracted. If the type is 'aligned' then the images are assumed to have
                     been generated with the reorient_images function from this module, or
                     celltool.contour.contour_tools.transform_image_to_contour, which that
                     function calls.
             - contour_match: if 'name' then contour files are matched to the image
                     names by looking for matching filenames (sans directories and extensions).
                     If 'order', then the contours are matched to the image names by their
                     order in the list.
             - image_names: a list of filenames of the images associated with the contours.
        """
        if image_type not in ('original', 'aligned'):
            raise RuntimeError("Image type %s is invalid. Must be 'original' or 'aligned'."%image_type)
        if contour_match not in ('name', 'order'):
            raise RuntimeError("Contour match method %s is invalid. Must be one of 'name' or 'order'."%contour_match)

        self.measurement_name = measurement_name
        _ImageMeasurementBase.__init__(self, image_type, contour_match, image_names)

    def header(self, contours):
        return [self.measurement_name]

    def measure(self, contour):
        image_name = self._get_image_for_contour(contour)
        image_array = _caching_reader.load(image_name)
        if self.image_type == 'original':
            contour = contour.as_world()
            c_min, c_max = contour.bounding_box()
            c_min = numpy.floor(c_min).astype(int)
            c_max = numpy.ceil(c_max).astype(int)
            domain = [c_min[0], c_min[1], c_max[0], c_max[1]]
            size = c_max - c_min
            mask = contour_tools.get_binary_mask(contour, domain=domain, size=size)
            image_array = image_array[c_min[0]:c_max[0], c_min[1]:c_max[1]]
        else:
            contour = contour.as_descaled()
            mask = contour_tools.get_binary_mask(contour, size=image_array.shape)
        return [image_array[mask.astype(bool)].sum()]

class _CachingImageReader(object):
    def __init__(self):
        self.cache = {}
    def prime(self, image_names):
        from celltool.utility import image
        self.image_module = image
        for name in image_names:
            value = self.cache.setdefault(name, [None, 0, 0])
            value[1] += 1
    def load(self, name):
        if name not in self.cache:
            return self.image_module.read_grayscale_array_from_image_file(name)
        else:
            image_array, max_loads, num_loads = self.cache[name]
            if image_array is None:
                image_array = self.image_module.read_grayscale_array_from_image_file(name)
            num_loads += 1
            if num_loads == max_loads:
                del(self.cache[name])
            else:
                self.cache[name] = [image_array, max_loads, num_loads]
            return image_array

_caching_reader = _CachingImageReader()