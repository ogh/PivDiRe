"""
The MIT License (MIT)

Copyright (c) 2014 Hubert Soyer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import warnings
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


def reduce_dim_2d(points_highd, pivot_pos=(), pivot_importance=5.0,
            lr=1e-3, stopping_threshold=1e-3, max_iter=300,
            metric='euclidean', seed=None):
    """
        Reduces the dimensionality of the given high-dimensional vectors.
        Instead of treating every point equally like PCA, SNE or tSNE,
        this method allows the user to specify a list of vectors in the
        given high dimensional data that should be treated with a preference.
        That means, distances between the specified "more important" pivot
        points and all other points will be preserved better.
        For now, this function supports only dimensionality reduction to 
        2d for plotting, it could, however, be easily extended 
        to more dimensions.
        
        :param points_highd: high dimensional vectors (n_vectors x dim)
        :param pivot_pos: list of indices corresponding to the vectors 
                            that should be weighed more
        :param pivot_importance: importance factor for pivot points.
                                    Higher means pivot-to-X distance will be
                                    preserved more. (default: 5.0)
        :param lr: learning rate for iterative optimization process 
                    (default: 1e-3)
        :param stopping_theshold: Stopping threshold for error difference
                                    between two optimization epochs
                                    (default: 1e-3)
        :param max_iter: maximal number of iterations (default: 300)
        :param metric: distance metrix on the _high dimensional_ points that
                        should be preserved as good as possible in 2d
        :param seed: random seed
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize the points
    points_2d = np.random.randn(points_highd.shape[0],2) * 0.01
    points_2d = (points_2d-points_2d.min()) / \
                        (points_2d.max()-points_2d.min()) * 2 - 1
    d_shape = (points_2d.shape[0],points_2d.shape[0],1)
    # Precompute the pairwise distances for the high dimensional vectors
    d_high = squareform(pdist(points_highd,metric=metric)).reshape(d_shape)

    points_highd = np.array(points_highd)
    last_err = np.infty
    threshold = 1e-3
    # Run the learning iterator
    for i in xrange(max_iter):
        
        # Compute the pairwise distances for the low dimensional vectors
        d_low = squareform(pdist(points_2d,metric='euclidean')).reshape(d_shape)
        # Get the pairwise differences between points for the low dimensional vectors
        diff_low = points_2d[:,None,:] - np.tile(points_2d,[points_2d.shape[0],1,1])
        # Compute the error (before the current update!)
        d_h_l = (d_high-d_low)
        error = (d_h_l**2)
        # Weight pivot points for error
        error[:,pivot_pos] *= pivot_importance
        error = error.sum()
        # Weight pivot points for gradient
        d_h_l[:,pivot_pos] *= pivot_importance
        # Compute gradient
        grad = ((-4) * d_h_l * diff_low).sum(axis=1)
        # Perform update
        points_2d -= lr * grad
        
        # Check of error between updates has fallen below
        # threshold
        if last_err - error < threshold and last_err > error:
            if i < 10:
                warnings.warn("Number of iterations before "
                    "hitting the error threshold was less than 10. "
                    "Results may not be satisfying, consider setting "
                    "a different threshold.")
            # If yes, break
            break
        last_err = error
        
    return points_2d 
