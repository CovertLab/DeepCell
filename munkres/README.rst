Munkres Readme
==============

Munkres calculates the minimum cost assignment of the assignment
using the Hungarian/Munkres algorithm.

Can handle non-square cost matricies, using algorithm
provded by Bougeois and Lassalle in `An extension of the Munkres
Algorithm for the Assignment Problem to Rectangular Matrices <http://dl.acm.org/citation.cfm?id=362945>`_.

Usage
=====
::

  from munkres import munkres
  import numpy as np
  a = np.array(map(float,'7 4 3 6 8 5 9 4 4'.split()), dtype=np.double).reshape((3,3))
  print munkres(a)

which should print out ::

 [[False False  True]
  [ True False False]
  [False  True False]]

.. image:: https://travis-ci.org/jfrelinger/cython-munkres-wrapper.png?branch=master
   :target: https://travis-ci.org/jfrelinger/cython-munkres-wrapper

