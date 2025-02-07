# Copyright 2007 Zachary Pincus
# This file is part of CellTool.
#
# CellTool is free software; you can redistribute it and/or modify
# it under the terms of version 2 of the GNU General Public License as
# published by the Free Software Foundation.

from . import contour_class
from celltool.numerics import utility_tools
import numpy

def contours_from_image(image_array, contour_value = None, closed_only = True, min_area = None, max_area = None, axis_align = False):
    """Find the contours at a given image intensity level from an image.
    If multiple contours are found, they are returned in order of increasing area.

    Parameters:
        - image_array: array object
        - contour_value: intensity level at which to extract contours. If None,
                then the intensity mid-point of the image is used.
        - closed_only: if True, then contours which touch the image edge (and thus
                are not cyclic) are discarded.
        - min_area, max_area: Minimum and maximum area (in pixels) of returned
                contours. Others will be discarded.
        - axis_align: if True, each contour will be aligned along its major axis.
    """
    from skimage import measure
    if contour_value is None:
        contour_value = numpy.ptp(image_array) / 2.0 + image_array.min()
    contour_points = measure.find_contours(image_array, contour_value)
    if closed_only:
        contour_points = [p for p in contour_points if numpy.allclose(p[-1], p[0])]

    contours = [contour_class.Contour(points = p, units = 'pixels') for p in contour_points]
    areas_and_contours = []
    for c in contours:
        area = c.signed_area()
        if area > 0:
            # Keep contours oriented in traditional (negative, counter-clockwise) orientation
            c.reverse_orientation()
            area = -area
        areas_and_contours.append((-area, c))
    if min_area is not None:
        areas_and_contours = [(a, c) for a, c in areas_and_contours if a >= min_area]
    if max_area is not None:
        areas_and_contours = [(a, c) for a, c in areas_and_contours if a <= max_area]
    if axis_align:
        for a, c in areas_and_contours:
            c.axis_align()
    areas_and_contours.sort(key=lambda ac: ac[0])
    return [c for a, c in areas_and_contours]

def _should_allow_reverse(contours, allow_reflection):
    # If we are to allow for reflections, we ought to allow for reversing
    # orientations too, because even if the contours start out oriented in the
    # same direction, reflections can change that.
    # Then check if all of the contours are oriented in the same direction.
    # If they're not, we need to allow for reversing their orientation in the
    # alignment process.
    if allow_reflection:
         return True
    orientations = numpy.array([numpy.sign(contour.signed_area()) for contour in contours])
    homogenous_orientations = numpy.all(orientations == -1) or numpy.all(orientations == 1)
    return not homogenous_orientations

def _compatibility_check(contours):
    if not utility_tools.all_same_shape([c.points for c in contours]):
        raise RuntimeError('All contours must have the same number of points in order to align them.')
    if numpy.all([isinstance(c, contour_class.ContourAndLandmarks) for c in contours]):
        # if they're all landmark'd contours
        all_landmarks = [c.landmarks for c in contours]
        if not utility_tools.all_same_shape(all_landmarks):
            raise RuntimeError('If all contours have landmarks, they must all have the same number of landmarks.')


def align_contour_to(contour, reference, global_align = True, align_steps = 8, allow_reflection = False,
        allow_scaling = False, weights = None, quick = False):
    """Optimally align a contour to a reference contour. The input contour will be
    transformed IN PLACE to reflect this alignment.

    Parameters:
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
        - allow_scaling: if True, then the contour may be scaled to fit the
                reference better.
        - weights: if provided, this must be a list of weights, one for each
                point, for weighting the fit between the contour and reference.
        - quick: if global_align is True and quick is True, then no local optimization
             will be performed at each of the global search steps. This will provide
             a rough and sub-optimal, but fast, alignment.

    See celltool.contour_class.Contour.global_best_alignment and local_best_alignment,
    which are used internally by this function, for more details.
    """
    _compatibility_check([contour, reference])
    allow_reversed_orientation = _should_allow_reverse([contour], allow_reflection)
    allow_translation = True
    if global_align:
        # axis-align first, so that the align_steps correspond to similar locations for each contour
        contour.axis_align()
        contour.global_best_alignment(reference, align_steps, weights, allow_reflection,
            allow_scaling, allow_translation, allow_reversed_orientation, quick)
    else:
        distance = contour.local_best_alignment(reference, weights, allow_reflection, allow_scaling, allow_translation)
        if allow_reversed_orientation:
            rev = self.as_reversed_orientation()
            r_distance = rev.local_best_alignment(reference, weights, allow_reflection, allow_scaling, allow_translation)
            if r_distance < distance:
                contour.__init__(other = rev)

def align_contours(contours, align_steps = 8, allow_reflection = False,
        allow_scaling = False, weights = None, max_iters = 10, min_rms_change = None,
        quick = False, iteration_callback = None):
    """Mutually align a set of contours to their mean in an expectation-maximization
    fashion. The input contous will be transformed IN PLACE to reflect this alignment.

    For each iteration, the mean contour is calculated, and then each contour is
    globally aligned to that mean with the celltool.contour_class.Contour.global_best_alignment
    method. Iteration continues until no contours are changed (beyond a given
    threshold), or the maximum number of iterations elapses.

    Parameters:
        - align_steps: The number of different contour orientations to consider
                when aligning each contour to the mean. For example, if align_steps = 8,
                then eight different (evenly-spaced) points will be chosen as the
                'first point' of the given contour, and then the fit to the mean
                will be locally optimized from that position. The best local fit is
                then treated as the global alignment.
        - allow_reflection: if True, then reflective transforms will be used if
                they make the alignment between the contour and reference better.
        - allow_scaling: if True, then the contour may be scaled to fit the
                reference better.
        - weights: if provided, this must be a list of weights, one for each
                point, for weighting the fit between the contour and reference.
        - max_iters: maximum number of alignment iterations.
        - min_rms_change: minimum RMS change between the contour points before
                and after alignment to the mean for that contour to be considered to
                have "changed". If no contours change, then iteration terminates;
                thus too stringent a criteria can prolong iteration, while too lax
                of one will produce sub-optimal results. If this parameter is None,
                then an appropriate value will be chosen.
        - quick: if True, then no local optimization will be performed at each of
             the global search steps. This will provide a rough and sub-optimal, but
             fast, alignment.
        - iteration_callback: if not None, this function is called after each
             contour is aligned, as follows: iteration_callback(iters, i, changed)
             where iters is the current iteration, i is the number of the contour
             that was just aligned, and changed is the number of contours changed
             so far during that iteration.

    See celltool.contour_class.Contour.global_best_alignment, which is used
    internally by this function, for more details.
    """

    _compatibility_check(contours)
    allow_reversed_orientation = _should_allow_reverse(contours, allow_reflection)
    allow_translation = True
    # roughly align the contours and make the point orderings correspond so that
    # the initial mean will be at all reasonable.
    for c in contours:
        c.axis_align()
        c.global_reorder_points(reference = contours[0])
    mean = contour_class.calculate_mean_contour(contours)
    if min_rms_change is None:
        # set the min RMSD to 0.01 of the largest dimension of the mean shape.
        min_rms_change = 0.01 * mean.size().max()
    min_ms_change = min_rms_change**2
    changed = 1
    iters = 0
    while changed != 0 and iters < max_iters:
        changed = 0
        for i, contour in enumerate(contours):
            original_points = contour.points[:]
            contour.global_best_alignment(mean, align_steps, weights, allow_reflection,
                allow_scaling, allow_translation, allow_reversed_orientation, quick)
            ms_change = ((contour.points - original_points)**2).mean()
            if ms_change > min_ms_change:
                changed += 1
            if iteration_callback is not None:
                iteration_callback(iters, i, changed)
        iters += 1
        mean = contour_class.calculate_mean_contour(contours)
    return iters


def get_binary_mask(contour, size, domain = None):
    """Get a binary mask of the given contour at a given (x-pixels, y-pixels)
    size. The spatial domain in terms of contour points which will be mapped
    to this grid can be specified as (x_min, y_min, x_max, y_max) in the
    'domain' parameter. If this parameter is 'None' then the spatial domain is
    set to the same size as the 'size' parameter (which implicitly assumes that
    the contour is scaled in pixel units), and centered about the middle point
    of the contour. (Not the geometric centroid, but the middle of the bounding
    box.)

    The returned binary mask is 0 outside the contour and 1 inside.
    """
    raise NotImplementedError() # TODO: compute with AGG
    if domain is None:
        # make a domain of the given size centered on the contour
        domain = numpy.array([[0, 0,], [size[0], size[1]]])
        center = numpy.array(size, dtype = float) / 2
        domain += contour.bounds_center() - center
    else:
        domain = numpy.asarray(domain)
    mask = closest_point_transform.mask_2d(contour.points, domain = domain.ravel(), samples = size)
    if contour.signed_area() > 0:
        # inside values are 0 and outside values are 1. Must reverse this.
        return 1 - mask
    else:
        return mask


def transform_image_to_contour(contour, image_array, size = None, mask = False):
    """Transform an image to be in the reference frame of a given contour.

    If a contour has been transformed after it has been extracted from an image
    (e.g. by alignment), this function can be used to trim the image region
    corresponding to that contour from an image and transform it by the same
    geometric transform that the contour has undergone.

    The 'image_array' parameter should be an array containing the original image,
    and 'size' should be an (x-pixels, y-pixels) pair. If 'size' is None, then
    the smallest size that completely contains the contour will be used.
    If 'mask' is True, then the image region outside of the contour is zeroed out.

    WARNING: If a contour has been scaled as part of its geometric transform,
    this function will ignore the scaling. Thus, the image will be transformed
    to the same orientation as the contour, but will still be in pixel units, not
    in the units that the contour has been scaled to. In most cases, this is the
    desired behavior.
    """
    from scipy import ndimage
    # Put the contour into pixel units so that when we transform the image,
    # it isn't resized up or down to whatever the contour units are...
    contour = contour.as_descaled()
    if size is None:
        size = numpy.ceil(contour.size()).astype(int)
    # now center the contour bounding box in the middle of the output image.
    center = numpy.array(size, dtype = float) / 2
    contour.recenter_bounds(center)
    # note that we take the to_world_transform because we need the one that goes from
    # output coordinates to input coordinates.
    # transpose the transform so that it's appropriate for column-vectors (the standard),
    # not row-vectors (what the contour uses).
    transform = contour.to_world_transform[:2,:2].transpose()
    offset = contour.to_world_transform[2,:2]
    transformed = ndimage.affine_transform(image_array, transform, offset, size, order=1)
    if mask:
        transformed *= get_binary_mask(contour, size)
    return transformed.astype(image_array.dtype)

def get_image_swath(contour, image_array, begin, end, offset, depth, l_samples = None, d_samples = None, image_type = 'original'):
    """Warp an image region into a rectangular "swath".

    One dimension of the warped region is defined by a contour, from contour point
    'begin' to    point 'end' (inclusive). Along this length, 'l_samples' points will
    be taken. The other dimension is defined by 'depth', which is a distance
    inward from the contour to be sample along at each point along the contour
    length. 'd_samples' different points will be taken along this dimension. The
    'offset' parameter controls where the sampling starts; negative values indicate
    that the region should begin outside of the contour and go inward, while positive
    values indicate that the region should start some distance inward already.

    The output swath is a masked array of dimension l_samples x d_samples, where
    the mask is 'True' if the point was outside of the original image. This array
    corresponds to the region from 'begin' to 'end' in one direction, and from
    the contour edge (plus 'offset'), inward 'depth' units.

    If the 'image_type' parameter is 'original' then the image array is assumed
    to be congruent to the image from which the contour was extracted. If this
    parameter is 'aligned', then the image array is assumed to have been generated
    with transform_image_to_contour; that is, the image corresponds to the contour,
    after the contour has been descaled to pixel units and centered on the image
    (but no other geometric transforms).

    If the ending point is less than the beginning point, the swath will wrap
    around the contour; it will not go in reverse contour-point-order.
    """
    l = len(contour.points)
    if begin is None: begin = 0
    if end is None: end = 0
    if end <= begin:
        end += l
    if l_samples is None:
        l_samples = end - begin
    if d_samples is None:
        d_samples = depth
    def position_getter(contour):
        l_points = numpy.linspace(begin, end, l_samples, endpoint=True)
        l_points %= l
        inward_normals = contour.inward_normals(l_points)
        offsets = numpy.linspace(offset, offset + depth, d_samples, endpoint = True)
        return numpy.multiply.outer(offsets, inward_normals) + contour.interpolate_points(l_points)
    return _map_contour_coords_to_image(contour, image_array, position_getter, image_type)

def get_rectangle_axis_swath(contour, image_array, depth, l_samples = None, d_samples = None, image_type = 'original'):
    """Warp a region around the contour's central axis into a rectangular swath.

     One dimension of the warped region is defined by the contour's central
    axis. Along this length, 'l_samples' points will be taken. The other
    dimension is defined by 'depth', which is a distance away from the central
    axis (on either side) that the swath will encompass. 'd_samples' different
    points will be taken along this dimension.

     The output swath is a masked array of dimension l_samples x d_samples,
    where the mask is 'True' if the point was outside of the original image.

     If the 'image_type' parameter is 'original' then the image array is assumed
    to be congruent to the image from which the contour was extracted. If this
    parameter is 'aligned', then the image array is assumed to have been
    generated with transform_image_to_contour; that is, the image corresponds to
    the contour, after the contour has been descaled to pixel units and centered
    on the image (but no other geometric transforms).
    """
    if d_samples is None:
        d_samples = depth
    if l_samples is None:
        l_samples = len(contour.central_axis)
    def position_getter(contour):
        l_points = numpy.linspace(0, len(contour.central_axis), l_samples, endpoint=True)
        inward_normals = contour.axis_normals()
        offsets = numpy.linspace(-depth, depth, d_samples, endpoint = True)
        return numpy.multiply.outer(offsets, inward_normals) + contour.interpolate_axis_points(l_points)
    return _map_contour_coords_to_image(contour, image_array, position_getter, image_type)

def get_axis_swath(contour, image_array, d_samples, begin=None, end=None, l_samples=None, image_type='original'):
    """Warp a central-axis contour into a rectangular swath.

     One dimension of the warped region is defined by the contour's central
    axis. Along this length, 'l_samples' points will be taken. The other
    dimension is defined by the top and bottom points along the contour.
    'd_samples' different points will be taken along this dimension.

     The output swath is a masked array of dimension l_samples x d_samples,
    where the mask is 'True' if the point was outside of the original image.

     If the 'image_type' parameter is 'original' then the image array is assumed
    to be congruent to the image from which the contour was extracted. If this
    parameter is 'aligned', then the image array is assumed to have been
    generated with transform_image_to_contour; that is, the image corresponds to
    the contour, after the contour has been descaled to pixel units and centered
    on the image (but no other geometric transforms).
    """
    l = len(contour.central_axis)
    if begin is None: begin = 0
    if end is None: end = l-1
    if l_samples is None:
        l_samples = end - begin + 1
    def position_getter(contour):
        from scipy.interpolate import fitpack
        l_points = numpy.linspace(begin, end, l_samples, endpoint=True)
        top_spline, bottom_spline = contour.axis_top_bottom_to_spline()
        c_points = len(contour.points)
        top_params = fitpack.splev(l_points, top_spline) % c_points
        bottom_params = fitpack.splev(l_points, bottom_spline) %c_points
        contour_spline, uout = contour.to_spline()
        top_points = numpy.transpose(fitpack.splev(top_params, contour_spline))
        bottom_points = numpy.transpose(fitpack.splev(bottom_params, contour_spline))
        mesh_points = numpy.empty((d_samples, l_samples, 2))
        interp_points = numpy.linspace(0, 1, d_samples)
        for i, ((tx, ty), (bx, by)) in enumerate(zip(top_points, bottom_points)):
            mesh_points[:,i,0] = numpy.interp(interp_points, [0, 1], [tx, bx])
            mesh_points[:,i,1] = numpy.interp(interp_points, [0, 1], [ty, by])
        return mesh_points
    return _map_contour_coords_to_image(contour, image_array, position_getter, image_type)

def _map_contour_coords_to_image(contour, image_array, position_getter, image_type):
    from scipy import ndimage
    if image_type not in ('original', 'aligned'):
        raise RuntimeError("Image type %s is invalid. Must be 'original' or 'aligned'."%image_type)
    if image_type == 'aligned':
        # we need to assume that the contour is centered on the origin, because
        # later we'll be scaling the points, and we don't want to deal with
        # the fact that the contour translation would be scaled too, if it's not
        # centered
        contour = contour.as_recentered_bounds()
    positions = position_getter(contour)
    # now transform the positions
    if image_type == 'original':
        # find the locations of the points on the original image
        transform = contour.to_world_transform
    else:
        # find the location of the points on an image in pixel coordinates
        # where the contour is centered on the image.
        transform = _get_descale_transform(contour, image_array.shape)
    shape = positions.shape
    positions = positions.reshape((shape[0]*shape[1], shape[2]))
    positions = utility_tools.homogenous_transform_points(positions, transform)
    positions = positions.reshape(shape)
    positions = positions.transpose((2,1,0))
    mapped = ndimage.map_coordinates(image_array, positions, output=float, cval=numpy.nan, order=1)
    return numpy.ma.array(mapped, mask = numpy.isnan(mapped))

def _get_descale_transform(contour, size, descale_only = True):
    """Get the transform that centers the contour's bounding box on an image with
     a given size, after descaling the contour to be in pixel units."""
    new_bounds_center = numpy.asarray(size, dtype=float) / 2
    old_bounds_center = contour.bounds_center()
    rotate_reflect, scale_shear, world_translation = utility_tools.decompose_homogenous_transform(contour.to_world_transform)
    old_bounds_center = numpy.dot([old_bounds_center], scale_shear)[0]
    translation = new_bounds_center - old_bounds_center
    return utility_tools.make_homogenous_transform(transform=scale_shear, translation=translation)

def add_image_landmarks(contour, image_array, landmark_ranges, image_type = 'original', mask = True):
    """Add landmarks to a contour (creating a ContourAndLandmarks object).

    The landmarks are defined from pixel intensities in an image: the 'landmark_ranges'
    parameter is a list of (low, high) tuples, where the position of a given landmark
    is taken as the geometric centroid of all of the pixels in the image falling
    within the intensity range of [low, high] (inclusive). This makes it easy for
    a user to 'paint' landmarks onto an image with particular intensities and then
    extract the centers of those landmarks. If the 'mask' parameter is true, then
    only pixels within the contour will be considered, which facilitates painting
    multiple landmarks on a single image from which multiple contours have derived.

    If the 'image_type' parameter is 'original' then the image array is assumed
    to be congruent to the image from which the contour was extracted. If this
    parameter is 'aligned', then the image array is assumed to have been generated
    with transform_image_to_contour; that is, the image corresponds to the contour,
    after the contour has been descaled to pixel units and centered on the image
    (but no other geometric transforms).
    """
    image_array = numpy.asarray(image_array, dtype=float)
    if image_type not in ('original', 'aligned'):
        raise RuntimeError("Image type %s is invalid. Must be 'original' or 'aligned'."%image_type)
    if image_type == 'original' and mask:
        # extract only the image region that we need to consider
        image_array = transform_image_to_contour(contour, image_array, mask = False)
        # now the image_array is aligned to the contour
        image_type = 'aligned'
    if image_type == 'original':
        to_image_transform = contour.to_world_transform
    else:
        # we're just interested in the transform that maps the contour to the middle
        # of the provided image (possibly descaling the contour back to pixel units)
        to_image_transform = _get_descale_transform(contour, image_array.shape)
    to_contour_transform = numpy.linalg.inv(to_image_transform)
    if mask:
        # if we need to mask the image, do so with nan's so that we can use any value
        # for the landmarks
        transformed_contour = contour.as_transformed(to_image_transform)
        domain = numpy.array([[0, 0], image_array.shape])
        image_mask = get_binary_mask(transformed_contour, image_array.shape, domain)
        nan_mask = numpy.where(image_mask, 1, numpy.nan)
        err = numpy.seterr(invalid='ignore')
        image_array *= nan_mask
        numpy.seterr(**err)
    landmarks = []
    for low, high in landmark_ranges:
        # a bit of fudge below to allow for some floating-point error to creep in
        # via the resampling that occurs when image_type is original and mask is true
        fudge = 1e-5
        indices = numpy.transpose(numpy.nonzero(numpy.logical_and(image_array>=(low-fudge), image_array<=(high+fudge))))
        if len(indices) == 0:
            if low == high:
                raise RuntimeError('No pixels at intensity value %d found for contour %s.'%(low, contour.simple_name()))
            else:
                raise RuntimeError('No pixels within intensity range [%d,%d] found for contour %s.'%(low, high, contour.simple_name()))
        landmarks.append(indices.mean(axis=0))
    landmarks = utility_tools.homogenous_transform_points(landmarks, to_contour_transform)
    if isinstance(contour, contour_class.ContourAndLandmarks):
        landmarks = numpy.concatenate(contour.landmarks, landmarks)
    landmark_contour = contour_class.ContourAndLandmarks(other=contour, landmarks=landmarks,
        weights = numpy.ones(len(contour.points)+len(landmarks)))
    return landmark_contour

def warp_images(from_contour, to_contour, image_arrays, output_region = None,
        from_type = 'original', to_type = 'original', interpolation_order = 1, approximate_grid = 1):
    """Define a thin-plate-spline warping transform that warps from the points of
    from_contour to the points of to_contour (and their landmarks, if they have any),
    and then warp the given images by that transform. In general, 'from_contour'
    should represent the shape of an object in the input images, and 'to_contour'
    should represent the desired shape after the warping. For example, this could
    be used to warp all shapes from multiple images to a single, canonical shape.

    Parameters:
        - from_contour and to_contour: instances of one of the classes defined in
                celltool.contour.contour_class that have corresponding points.
        - image_arrays: list of images to warp with the given warp transform.
        - output_region: the (xmin, ymin, xmax, ymax) region of the output
                image that should be produced.    If the to_type is 'original'; then a
                specific sub-region of the image corresponding to the to_contour can be
                selected. If to_type is 'aligned', then this just selects the size of
                the output images, and should be specified as (0, 0, x-pixels, y-pixels).
        - from_type: the type of images that are passed in the 'image_arrays' parameter.
                If 'original' then the images are assumed to be congruent to the image
                from which the from_contour was originally extracted; if 'aligned' then
                the images are assumed to have been produced via the
                transform_image_to_contour function.
        - to_type: if 'original' then the warp transforms the images into the frame
                of reference of the original contour, in its location in the image from
                which it was extracted. If 'aligned' then warp the images to the current
                orientation of the to_contour.
        - interpolation_order: if 1, then use linear interpolation; if 0 then use
                nearest-neighbor.
        - approximate_grid: defining the warping transform is slow. If approximate_grid
                is greater than 1, then the transform is defined on a grid 'approximate_grid'
                times smaller than the output image region, and then the transform is
                bilinearly interpolated to the larger region. This is fairly accurate
                for values up to 10 or so.
    """
    from celltool.numerics import image_warp
    _compatibility_check([from_contour, to_contour])
    image_arrays = [numpy.asarray(image_array) for image_array in image_arrays]
    if not utility_tools.all_same_shape(image_arrays):
        raise ValueError('Input images for warp_images must all be the same size.')
    if from_type not in ('original', 'aligned') or to_type not in ('original', 'aligned'):
        raise RuntimeError("Image from_type or to_type %s is invalid. Must be 'original' or 'aligned'."%image_type)
    if output_region is None:
        output_region = [0, 0, image_arrays[0].shape[0], image_arrays[0].shape[1]]
    if to_type == 'original':
        # put the to_contour in world units
        to_contour = to_contour.as_world()
    else:
        # center the to_contour bounding box in the middle of the output images, and
        # descale it to be in pixel units
        to_contour = to_contour.as_descaled()
        center = numpy.ptp(numpy.array([output_region[:2],output_region[2:]], dtype = float), axis=0) / 2
        to_contour.recenter_bounds(center)
    if from_type == 'original':
        # put the from_contour in world units
        from_contour = from_contour.as_world()
    else:
        # center the from_contour bounding box in the middle of the input image, and
        # descale it to be in pixel units
        from_contour = from_contour.as_descaled()
        center = numpy.array(image_arrays[0].shape, dtype = float) / 2
        from_contour.recenter_bounds(center)
    if isinstance(from_contour, contour_class.ContourAndLandmarks) and isinstance(to_contour, contour_class.ContourAndLandmarks):
        # in case the contours have landmarks, ask them to pack the landmarks into
        # their points list
        from_contour._pack_landmarks_into_points()
        to_contour._pack_landmarks_into_points()
    return image_warp.warp_images(from_contour.points, to_contour.points, image_arrays,
        output_region, interpolation_order, approximate_grid)
