#!/usr/bin/env python3

'''
test funcs from differential_drive_model
'''
from __future__ import print_function

import numpy as np
import rospy

import differential_drive_model as ddm

def test_get_odometry_covariance():
  """

  :return:
  """
  addm = ddm.DifferentialDriveModel(0.03, 0.01, 0.04, 0.0076)
  noise_cov = addm.get_process_noise_covariance([1, 1, 0])
  expected_cov = np.array([[0.03**2, 0, 0],
                           [0, 0.01**2, 0],
                           [0, 0, 0.03**2*4.01**2]])
  assert np.allclose(expected_cov, noise_cov)

  timed_T_WV_list = [ # pylint: disable=invalid-name
      [rospy.Time(1.0), 1, 0, 0, 0, 0, 0, 1],
      [rospy.Time(2.0), 2, 0, 0, 0, 0, 0, 1],
      [rospy.Time(3.0), 3, 0, 0, 0, 0, 0, 1],
      [rospy.Time(4.0), 4, 0, 0, 0, 0, 0, 1],
  ]
  P = addm.get_odometry_covariance(timed_T_WV_list) # pylint: disable=invalid-name
  exp_P = np.array([[0.0027, 0., 0.], # pylint: disable=invalid-name
                    [0., 0.07236045, 0.04341627],
                    [0., 0.04341627, 0.04341627]])
  assert np.allclose(P, exp_P)

