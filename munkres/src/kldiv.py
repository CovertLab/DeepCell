from __future__ import division
import numpy as np
from numpy.linalg import det, inv

def true_kldiv(m0, m1, s0, s1):
    k = m0.shape[0]
    first = np.trace(np.dot(inv(s1), s0))
    mdiff = m1 - m0
    second = np.dot(mdiff.T, np.dot(inv(s1), mdiff))
    third = np.log(det(s0) / det(s1))

    return .5 * (first + second - third - k)

def true_skldiv(m0, m1, s0, s1):
    return true_kldiv(m0, m1, s0, s1) + true_kldiv(m1, m0, s1, s0)
