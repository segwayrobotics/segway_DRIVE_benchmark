#!/usr/bin/env python

"""
test functions from tf_helpers
"""

import tf_helpers
import numpy as np


def test_transformtransformation():
  """
  test func
  :return:
  """
  mat = np.array(
      [[1.00000000e+00, -1.07565694e-15, -1.55732588e-15, 1.11022302e-15],
       [-6.03453836e-16, 1.00000000e+00, -3.48313117e-16, 5.55111512e-16],
       [-2.66448070e-16, 3.85094971e-16, 1.00000000e+00, 1.38777878e-17],
       [-3.53332323e-16, 3.11761750e-16, - 3.59784738e-16, 1.00000000e+00]]
  )
  vec = tf_helpers.transformtransformation(mat)
  expected_vec = [0.0] * 7
  expected_vec[6] = 1.0
  assert np.allclose(expected_vec, vec)
  dist, angle = tf_helpers.measure_difference(vec)
  assert np.allclose(dist, 0.0) and np.allclose(angle, 0.0)


def test_measure_difference():
  """
  test func
  :return:
  """
  delta_vec = [
      1.1102230246251565e-15, 5.5511151231257827e-16, 1.3877787807814457e-17,
      1.8335202208441998e-16, -3.2271945275451216e-16, 1.1805077592635108e-16,
      1.0000000000000002]
  dist, angle = tf_helpers.measure_difference(delta_vec)
  assert np.allclose(dist, 0.0) and np.allclose(angle, 0.0)
