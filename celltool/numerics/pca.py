# Copyright 2007 Zachary Pincus
# This file is part of CellTool.
# 
# CellTool is free software; you can redistribute it and/or modify
# it under the terms of version 2 of the GNU General Public License as
# published by the Free Software Foundation.

import bisect, numpy
import utility_tools

def pca(data):
    """pca(data, axis) -> mean, pcs, norm_pcs, variances, positions, norm_positions
    Perform Principal Components Analysis on a set of n-dimensional data points.
    The data array must be packed such that 'data[i]' is the ith data point.
    This function returns the mean data point, the principal components (packed 
    such that 'pcs[i]' is the ith principal component), the normalized
    principal components (each component is normalized by the data's standard
    deviation along that component), the variance each component represents, the 
    position of each data point along each component, and the position of each
    data point along each normalized component."""
    data = numpy.asarray(data)
    mean = data.mean(axis = 0)
    centered = data - mean
    flat, data_point_shape = utility_tools.flatten_data(centered)
    # could use _flat_pca_svd, but that appears empirically slower...
    pcs, variances, stds, positions, norm_positions = _flat_pca_eig(flat)    
    norm_pcs = utility_tools.fatten_data(pcs * stds[:, numpy.newaxis], data_point_shape)
    pcs = utility_tools.fatten_data(pcs, data_point_shape)
    return mean, pcs, norm_pcs, variances, positions, norm_positions
    
def _flat_pca_svd(flat):
    u, s, vt = numpy.linalg.svd(flat, full_matrices = 0)
    pcs = vt
    v = numpy.transpose(vt)
    data_count = len(flat)
    variances = s**2 / data_count
    root_data_count = numpy.sqrt(data_count)
    stds = s / root_data_count
    positions =    u * s
    norm_positions = u * root_data_count
    return pcs, variances, stds, positions, norm_positions

def _flat_pca_eig(flat):
    values, vectors = _symm_eig(flat)
    pcs = vectors.transpose()
    variances = values / len(flat)
    stds = numpy.sqrt(variances)
    positions = numpy.dot(flat, vectors)
    err = numpy.seterr(divide='ignore', invalid='ignore')
    norm_positions = positions / stds
    numpy.seterr(**err)
    norm_positions = numpy.where(numpy.isfinite(norm_positions), norm_positions, 0)
    return pcs, variances, stds, positions, norm_positions

def _symm_eig(a):
    """Return the eigenvectors and eigenvalues of the symmetric matrix a'a. If
    a has more columns than rows, then that matrix will be rank-deficient,
    and the non-zero eigenvalues and eigenvectors can be more easily extracted
    from the matrix aa', from the properties of the SVD:
        if a of shape (m,n) has SVD u*s*v', then:
            a'a = v*s's*v'
            aa' = u*ss'*u'
        let s_hat, an array of shape (m,n), be such that s * s_hat = I(m,m) 
        and s_hat * s = I(n,n). Thus, we can solve for u or v in terms of the other:
            v = a'*u*s_hat'
            u = a*v*s_hat            
    """
    m, n = a.shape
    if m >= n:
        # just return the eigenvalues and eigenvectors of a'a
        vecs, vals = _eigh(numpy.dot(a.transpose(), a))
        vecs = numpy.where(vecs < 0, 0, vecs)
        return vecs, vals
    else:
        # figure out the eigenvalues and vectors based on aa', which is smaller
        sst_diag, u = _eigh(numpy.dot(a, a.transpose()))
        # in case due to numerical instabilities we have sst_diag < 0 anywhere, 
        # peg them to zero
        sst_diag = numpy.where(sst_diag < 0, 0, sst_diag)
        # now get the inverse square root of the diagonal, which will form the
        # main diagonal of s_hat
        err = numpy.seterr(divide='ignore', invalid='ignore')
        s_hat_diag = 1/numpy.sqrt(sst_diag)
        numpy.seterr(**err)
        s_hat_diag = numpy.where(numpy.isfinite(s_hat_diag), s_hat_diag, 0)
        # s_hat_diag is a list of length m, a'u is (n,m), so we can just use
        # numpy's broadcasting instead of matrix multiplication, and only create
        # the upper mxm block of a'u, since that's all we'll use anyway...
        v = numpy.dot(a.transpose(), u[:,:m]) * s_hat_diag
        return sst_diag, v

def _eigh(m):
    values, vectors = numpy.linalg.eigh(m)
    order = numpy.flipud(values.argsort())
    return values[order], vectors[:,order]

def pca_dimensionality_reduce(data, required_variance_explained):
    mean, pcs, norm_pcs, variances, positions, norm_positions = pca(data)
    total_variance = numpy.add.accumulate(variances / numpy.sum(variances))
    num = bisect.bisect(total_variance, required_variance_explained) + 1
    return mean, pcs[:num], norm_pcs[:num], variances[:num], numpy.sum(variances), positions[:,:num], norm_positions[:,:num]

def pca_reconstruct(scores, pcs, mean):
    # scores and pcs are indexed along axis zero
    flat, data_point_shape = utility_tools.flatten_data(pcs)
    return mean + utility_tools.fatten_data(numpy.dot(scores, flat), data_point_shape)

def pca_decompose(data, pcs, mean, variances = None):
    flat_pcs, data_point_shape = utility_tools.flatten_data(pcs)
    flat_data, data_point_shape = utility_tools.flatten_data(data - mean)
    projection = numpy.dot(flat_data, flat_pcs.transpose())
    if variances is not None:
        normalized_projection = projection / numpy.sqrt(variances)
        return projection, normalized_projection
    else:
        return projection