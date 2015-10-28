# Copyright 2007 Zachary Pincus
# This file is part of CellTool.
# 
# CellTool is free software; you can redistribute it and/or modify
# it under the terms of version 2 of the GNU General Public License as
# published by the Free Software Foundation.

def procustes_alignment(points, reference, weights = None, allow_reflection = False, find_scale = True, find_translation = True):
    """Find the rigid transformation that optimally aligns the given points to the
    reference points in a least-squares sense. The 'points' and 'references' parameters
    should both be NxD arrays of N corresponding D-dimensional points.
    
    The return value is a tuple of (transformation, scale, translation, new_points),
    where 'transformation' is a rotation matrix, 'scale' is a (scalar) scaling factor,
    'translation' is a translation vector, and new_points is the image of the 
    points under this transformation.
    
    Parameters:
        - weights: if not None, the least-squares fit is weighted by these values.
        - allow_reflection: if True, then the 'transformation' matrix may be a 
                rotation or rotation-and-reflection, depending in which results in a 
                better fit.
        - find_scale: if True, then the 'scale' value may be other than unity if
                scaling the points results in a better fit.
        - find_translation: if True, then the 'translation' vector may be other 
                than a zero-vector, if translating the points results in a better fit.
    """
    
    # notational conventions after "Generalized Procrustes Analysis and its Applications in Photogrammetry"
    # by Devrim Akca
    import numpy.matlib
    A = numpy.matrix(points)
    B = numpy.matrix(reference)
    if points.shape != reference.shape:
        raise TypeError('Cannot align point-sets in different dimensions, or with different numbers of points.')
    p = A.shape[0]
    k = A.shape[1]
    j = numpy.matlib.ones((p, 1), dtype = float)
    if weights is not None:
        Q = numpy.matrix(numpy.sqrt(weights)).T
        # here use numpy-array element-wise multiplication with broadcasting:
        A = numpy.multiply(A, Q)
        B = numpy.multiply(B, Q)
        j = numpy.multiply(j, Q)
    jjt = j * j.T
    jtj = j.T * j
    I = numpy.matlib.eye(p)
    At_prod = A.T * (I - jjt / jtj)
    S = At_prod * B
    V,D,Wt = numpy.linalg.svd(S)
    if not allow_reflection:
        if numpy.allclose(numpy.linalg.det(V), -1):
            V[:, -1] *= -1
        if numpy.allclose(numpy.linalg.det(Wt), -1):
            Wt[-1, :] *= -1
    T = numpy.dot(V, Wt)
    if find_scale:
        c = numpy.trace(T.T * S) / numpy.trace(At_prod * A)
    else:
        c = 1
    new_A = c * A * T
    if find_translation:
        t = (B - new_A).T * (j / jtj)
        # now unpack t from a 2d matrix-vector into a normal numpy 1d array-vector
        t = numpy.asarray(t)[:,0]
    else:
        t = numpy.zeros(k)
    if weights is not None:
        new_A = numpy.divide(new_A, Q)
    new_A += t
    return numpy.asarray(T), c, t, numpy.asarray(new_A)