# Copyright 2007 Zachary Pincus
# This file is part of CellTool.
#
# CellTool is free software; you can redistribute it and/or modify
# it under the terms of version 2 of the GNU General Public License as
# published by the Free Software Foundation.

import numpy

def distance(a, b):
    """Return the euclidian distance between two vectors."""
    return numpy.sqrt(distance_squared(a, b))

def distance_squared(a, b):
    """Return the squared distance between two vectors."""
    a = numpy.asarray(a)
    b = numpy.asarray(b)
    return ((a - b)**2).sum()

def squared_distance_matrix(x):
    """Returns the matrix of pairwise distances between elements of x.
    (Note: x must be at least two-dimensional. x=[1,2,3,4] will not work, but
    x=[[1],[2],[3],[4]] will.)

    The returned matrix is of shape (len(x), len(x)), but only the upper
    triangle is filled in, so distances[i, j] returns a correct value only if
    i <= j. Other entries are zero.
    """
    x = numpy.asarray(x)
    l = len(x)
    distances = numpy.zeros((l, l), dtype=float)
    # distances = numpy.empty((l, l), dtype=float)
    # diag = numpy.ndarray(buffer=distances, dtype=distances.dtype, shape=(l,), strides=((l+1)*distances.dtype.itemsize,))
    # diag[:] = 0
    for i in range(l-1):
        dists = ((x[i+1:] - x[i])**2).sum(axis=-1)
        distances[i, i+1:] = dists
        # distances[i+1:, i] = dists
    return distances


def norm(data, axis = 0):
    """Return the 2-norm of a set of data points indexed along the given axis."""
    data = numpy.asarray(data)
    if axis != 0: data = numpy.swapaxes(data, axis, 0)
    data, oldshape = flatten_data(data)
    return numpy.sqrt((data**2).sum(axis = 1))

def first_derivative(data):
    """Calculate the first derivative along the first axis via central differences
    except at the endpoints where the forward/backward difference is used."""
    data = numpy.asarray(data)
    out = numpy.empty(data.shape, dtype=float)
    out[1:-1] = (data[2:] - data[:-2])/2.0
    out[0] = (data[1] - data[0])
    out[-1] = (data[-1] - data[-2])
    return out

def periodic_first_derivative(data):
    """Calculate the first derivative of data via central differences, assuming that
    the data are periodic."""
    forward = numpy.roll(data, -1, axis = 0)
    backward = numpy.roll(data, 1, axis = 0)
    return (forward - backward) / 2

def periodic_second_derivative(data):
    """Calculate the second derivative of data via central differences, assuming that
    the data are periodic."""
    forward = numpy.roll(data, -1, axis = 0)
    backward = numpy.roll(data, 1, axis = 0)
    return (forward - 2 * data + backward)

def flatten_data(data):
    """Input is a matrix of d n-dimensional data points. Data points are
    assumed to be indexed by axis zero. E.g. data[i, ...] is the ith data point.
    Returns a 2-dimensional matrix where each data point has been flattened to
    a 1-D vector. Also returns the original shape of the data points."""
    data = numpy.asarray(data)
    data_count = data.shape[0]
    data_point_shape = data.shape[1:]
    if len(data.shape) > 1:
        flat = numpy.reshape(data, (data_count, numpy.prod(data_point_shape)))
    else:
        flat = numpy.atleast_2d(data).transpose()
    return flat, data_point_shape

def fatten_data(data, data_point_shape):
    """Oposite of flatten_data: takes a set of data points that have been packed
    as 1-D vectors and expands them to their original shape."""
    return numpy.reshape(data, [data.shape[0]] + list(data_point_shape))

def periodic_slice(data, begin = None, end = None):
    """Slice the given data array as if it were a circle. Slices where 'begin' is
    after 'end' just result in the data being wrapped around. The [begin, end)
    interval is half-open, as in python slices."""
    l = len(data)
    if begin is None: begin = 0
    else: begin %= l
    if end is None: end = l
    else: end %= l
    if end > begin:
        return data[begin:end]
    else:
        return numpy.concatenate([data[begin:], data[:end]])

def inclusive_periodic_slice(data, begin = None, end = None):
    """Return a periodic slice (see the 'periodic_slice' function) over the
    [begin, end] interval, inclusive."""
    if end is not None: end += 1
    return periodic_slice(data, begin, end)

def make_homogenous_transform(transform = [[1,0],[0,1]], scale = 1, translation = [0,0]):
    """Return a 3x3 homogenous transform (for row vectors) from a 2x2 rigid transform
    matrix, a scalar scale factor, and a translation vector."""
    T = numpy.zeros((3,3))
    T[:2, :2] = numpy.asarray(scale)*numpy.asarray(transform)
    T[2, :2] = numpy.asarray(translation)
    T[2,2] = 1
    return T

def decompose_homogenous_transform(transform):
    """Decompose a 3x3 homogenous transform (for row vectors) into a 2x2 rotation/reflection
    matrix, a 2x2 scale/shear matrix, and a translation vector."""
    translation = transform[2, :2]
    T = transform[:2, :2]
    # decompose transform T into rotation/reflection Q and scale/shear S via polar decomposition
    u, s, vt = numpy.linalg.svd(T)
    Q = numpy.dot(u, vt)
    S = numpy.dot(numpy.dot(vt.transpose(), numpy.diagflat(s)), vt)
    return Q, S, translation

def homogenous_transform_points(points, transform):
    """Transform a list of 2D points with a 3x3 homogenous transform which operates on row-vectors."""
    points = numpy.asarray(points)
    transform = numpy.asarray(transform)
    homogenous_points = numpy.ones((points.shape[0], 3))
    homogenous_points[:,:2] = points
    transformed_points = numpy.dot(homogenous_points, transform)
    transformed_points /= transformed_points[:,numpy.newaxis,2]
    return transformed_points[:,:2]

def all_same_shape(arrays):
    """Return True if all input arrays are the same shape"""
    shape = numpy.asarray(arrays[0]).shape
    for a in arrays[1:]:
        if shape != numpy.asarray(a).shape:
            return False
    return True

def b_spline_to_bezier_series(tck, per = False):
    """Convert a parametric b-spline into a sequence of Bezier curves of the same degree.

    Inputs:
        tck : (t,c,k) tuple of b-spline knots, coefficients, and degree returned by splprep.
        per : if tck was created as a periodic spline, per *must* be true, else per *must* be false.

    Output:
        A list of Bezier curves of degree k that is equivalent to the input spline.
        Each Bezier curve is an array of shape (k+1,d) where d is the dimension of the
        space; thus the curve includes the starting point, the k-1 internal control
        points, and the endpoint, where each point is of d dimensions.
    """
    from scipy.interpolate.fitpack import insert
    t,c,k = tck
    t = numpy.asarray(t)
    try:
        c[0][0]
    except:
        # I can't figure out a simple way to convert nonparametric splines to
        # parametric splines. Oh well.
        raise TypeError("Only parametric b-splines are supported.")
    new_tck = tck
    if per:
        # ignore the leading and trailing k knots that exist to enforce periodicity
        knots_to_consider = numpy.unique(t[k:-k])
    else:
        # the first and last k+1 knots are identical in the non-periodic case, so
        # no need to consider them when increasing the knot multiplicities below
        knots_to_consider = numpy.unique(t[k+1:-k-1])
    # For each unique knot, bring its multiplicity up to the next multiple of k+1
    # This removes all continuity constraints between each of the original knots,
    # creating a set of independent Bezier curves.
    desired_multiplicity = k+1
    for x in knots_to_consider:
        current_multiplicity = numpy.sum(t == x)
        remainder = current_multiplicity%desired_multiplicity
        if remainder != 0:
            # add enough knots to bring the current multiplicity up to the desired multiplicity
            number_to_insert = desired_multiplicity - remainder
            new_tck = insert(x, new_tck, number_to_insert, per)
    tt,cc,kk = new_tck
    # strip off the last k+1 knots, as they are redundant after knot insertion
    bezier_points = numpy.transpose(cc)[:-desired_multiplicity]
    if per:
        # again, ignore the leading and trailing k knots
        bezier_points = bezier_points[k:-k]
    # group the points into the desired bezier curves
    return numpy.split(bezier_points, len(bezier_points) / desired_multiplicity, axis = 0)

def line_intersections(start, end, ref_start, ref_end):
    """Determine the points of intersection of a line (specified by two (x, y)
    points, start and end) with a number of other reference lines (specified as
    lists of (x, y) starting points -- ref_start -- and ending points --
    ref_end).
    Two arrays are returned: line_fractions and ref_fractions. The first
    contains a list of fractional distances between start and end at which the
    line intersects a particular reference line (or nan if no intersection).
    Note that these distances could be positive or negative.
    The second array contains a list of the fractional distances between the
    start and end of the given reference line where the intersection occur, or
    nan if no intersection.)
    """
    p0 = numpy.asarray(start)
    p1 = numpy.asarray(end)
    q0 = numpy.asarray(ref_start)
    q1 = numpy.asarray(ref_end)
    u = p1 - p0
    v = q1 - q0
    w = p0 - q0
    err = numpy.seterr(divide='ignore')
    denom = v[:,0]*u[1] - v[:,1]*u[0]
    s_i = (v[:,1]*w[:,0] - v[:,0]*w[:,1])/denom
    t_i = (u[0]*w[:,1] - u[1]*w[:,0])/-denom
    numpy.seterr(**err)
    return s_i, t_i

def closest_point_to_lines(point, lines_start, lines_end):
    """Given a point and a set of lines (specified parametrically by starting
    and ending points), return the point on each line that is closest to the
    given point, and the parametric position along each line of that point."""
    v = lines_end - lines_start
    w = point - lines_start
    c1 = (v*w).sum(axis=1)
    c2 = (v*v).sum(axis=1)
    fractional_positions = c1 / c2
    closest_points = lines_start + fractional_positions[:,numpy.newaxis]*v
    return closest_points, fractional_positions

def closest_points_to_line(points, line_start, line_end):
    """Given a set of points and a line (specified parametrically by starting
    and ending points), return the points on the line that are closest to the
    given points, and the parametric position along the line of those points."""
    v = line_end - line_start
    w = points - line_start
    c1 = (v*w).sum(axis=1)
    c2 = (v*v).sum()
    fractional_positions = c1 / c2
    closest_points = line_start + numpy.multiply.outer(fractional_positions, v)
    return closest_points, fractional_positions

def signed_distances_to_line(points, line_start, line_end):
    """Return the distances from the given points to the specified parametric
    line. The distances are signed: positive is on one side of the line,
    negative the other."""
    x, y = points.T
    x0, y0 = line_start
    x1, y1 = line_end
    return ((y0 - y1)*x + (x1 - x0)*y + (x0*y1 - x1*y0)) / numpy.sqrt(((line_start-line_end)**2).sum())

def find_perp(p0, p1, normalize=True):
    """Find a vector perpendicular to the line p0-p1."""
    p1 = p1.astype(float)
    diff = p1 - p0
    diff = numpy.roll(diff, 1, axis=-1)
    diff[...,0] *= -1
    if normalize:
        diff /= numpy.sqrt(numpy.sum(diff**2, axis=-1))[...,numpy.newaxis]
    return diff

def find_bisector(p0, p1, p2):
    """Given three points that form an angle p0-p1-p2, find a point p3 whereby
    p0-p1-p3 or p2-p1-p3 is the bisector of that angle."""
    d1 = p1 - p0
    d2 = p2 - p1
    a1 = numpy.arctan2(d1[...,1], d1[...,0])
    a2 = numpy.arctan2(d2[...,1], d2[...,0])
    ad = a2 - a1
    af = a1 + (numpy.pi + ad) / 2
    return numpy.transpose([numpy.cos(af), numpy.sin(af)])


def parabola_estimate_center(x_values, y_values, index, cyclic=False):
    """Estimate the x-value of the center of the parabola defined by the
    points surrounding 'index'. If 'index' is at the edge of the array and
    it is not cyclic, then the x-value corresponding to the index is returned.
    """
    x_values = numpy.asarray(x_values)
    y_values = numpy.asarray(y_values)
    l = len(y_values)
    indices = numpy.array([index-1, index, index+1])
    if index == 0 or index == (l-1):
        if cyclic:
            indices %= l
        else:
            return x_values[index]
    x_values = x_values[indices]
    y_values = y_values[indices]
    A = numpy.ones((3,3), dtype=float)
    A[:,0] = x_values**2
    A[:,1] = x_values
    a, b, c = numpy.linalg.solve(A, y_values)
    center = -b / (2*a)
    return center

def local_max(array, index, cyclic=False):
    """Find the local maximum of an array by hill-climbing from the provided
    index.
    Returns the index and the current value at that index.
    In the event that it could hill-climb either direction, the bias is toward
    going left.
    """
    array = numpy.asarray(array)
    l = len(array)
    current = array[index]
    left_i = _index_at(l, index, -1, cyclic)
    right_i = _index_at(l, index, 1, cyclic)
    left = array[left_i]
    right = array[right_i]
    if current >= left and current >= right:
        return index, current
    elif current < left:
        direction = -1
        index = left_i
        current = left
    else:
        direction = 1
        index = right_i
        current = right
    while True:
        new_index = _index_at(l, index, direction, cyclic)
        if array[new_index] <= current:
            return index, current
        index = new_index
        current = array[index]

def _index_at(l, index, direction, cyclic):
    new_index = index + direction
    if cyclic:
        new_index %= l
    else:
        new_index = numpy.clip(new_index, 0, l-1)
    return new_index

def local_maxima(array, min_distance = 1, cyclic=False, endpoints_allowed=True):
    """Find all local maxima of the array, separated by at least min_distance."""
    from scipy import ndimage
    array = numpy.asarray(array)
    cval = 0
    if cyclic:
        mode = 'wrap'
    elif endpoints_allowed:
        mode = 'nearest'
    else:
        mode = 'constant'
        cval = array.max()+1
    return numpy.arange(len(array))[array == ndimage.maximum_filter(array, 1+2*min_distance, mode=mode, cval=cval)]

def A_star(start, goal, successors, edge_cost, heuristic_cost_to_goal=lambda position, goal:0):
    """Very general a-star search. Start and goal are objects to be compared
    with the '==' operator, successors is a function that, given a node, returns
    other nodes reachable therefrom, edge_cost is a function that returns the
    cost to travel between two nodes, and heuristic_cost_to_goal is an
    admissible heuristic function that gives an underestimate of the cost from a
    position to the goal."""
    import heapq
    closed = set()
    open = [(0, 0, (start,))]
    while open:
        heuristic_cost, cost_so_far, path = heapq.heappop(open)
        tail = path[-1]
        if tail in closed:
            continue
        if tail == goal:
            return path
        closed.add(tail)
        for new_tail in successors(tail):
            new_cost_so_far = cost_so_far + edge_cost(tail, new_tail)
            new_heuristic_cost = new_cost_so_far + heuristic_cost_to_goal(new_tail, goal)
            heapq.heappush(open, (new_cost_so_far, new_heuristic_cost, path+(new_tail,)))
    raise RuntimeError('No path found.')
