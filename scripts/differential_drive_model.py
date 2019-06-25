#!/usr/bin/env python3

'''
A differential drive model for the segway robots, adapted from the c++ version
use to get the covariance for the odometry
'''
from __future__ import print_function

import copy
import math
import numpy as np
import tf_helpers


class Control(object):
  """the controp input expressed by dx, dy, dtheta"""
  def __init__(self, xytheta, duration):
    """
    :param xytheta: dx, dy, dtheta
    :param duration: float duration in secs
    """
    self.rel_pose = xytheta # Vk_T_Vkp1, {V} z along the negative gravity
    self.delta_time = duration

  def to_delta(self, velocity_y_threshold, velocity_y_theta_ratio,
               angular_rate_threshold):
    """
    cap the transverse motion by heuristics
    :return: the control modified in the transverse direction
    """
    cap_transverse_vel = False
    if not cap_transverse_vel:
      return self.rel_pose

    yaw = self.rel_pose[2]
    trans = copy.deepcopy(self.rel_pose)

    velocity_y = trans[1] / self.delta_time
    velocity_theta = yaw / self.delta_time

    velocity_y_new = min(
        min(abs(velocity_y), velocity_y_threshold),
        abs(velocity_theta) * velocity_y_theta_ratio)
    deltay = velocity_y_new * self.delta_time
    # keep the direction info
    if velocity_y < 0:
      deltay = - deltay
    if abs(velocity_theta) > angular_rate_threshold:
    # if much rotation occurs, deltay can be large
      deltay = trans[1]
    return [trans[0], deltay, yaw]


class DifferentialDriveModel(object):
  """
  differential drive model referring to http://correll.cs.colorado.edu/?p=1307
      Introduction_to_Robotics_Lab_8_Error_Propagation_Correll_Lab.pdf and
      https://github.com/correll/Introduction-to-Autonomous-Robots/releases/download/v1.9.1/book_EN.pdf
      page 54-58
  a few heuristics are added to curb the transverse velocity and avoid
      degeneracy, with insights from page 139 sec 5.4 odometry motion model
      eq (5.44 - 5.46) of probabilistic robotics
  """
  def __init__(self, dxdx_ratio=0.03, dydy_ratio=0.01,
               init_cov_xy=0.1**2, init_cov_theta=(5*math.pi/180)**2):
    """
    init parameters for the model
    """
    self.dx_dx_drift_ratio = dxdx_ratio
    self.dy_dy_drift_ratio = dydy_ratio
    self.dist_angle_drift_ratio = dxdx_ratio*4.01
    self.angle_dist_drift_ratio = dxdx_ratio/4
    self.angle_angle_drift_ratio = dxdx_ratio

    self.loc_covariance_xy = init_cov_xy
    self.loc_covariance_theta = init_cov_theta
    self.velocity_y_threshold = 0.3
    self.velocity_y_theta_ratio = 0.3
    self.angular_rate_threshold = 0.3

    self.hokuyo_ref_cov_xy = 0.05 ** 2
    self.hokuyo_ref_cov_theta = (3*math.pi/180) ** 2


  def get_process_noise_covariance(self, dxytheta):
    """
    compute covariance for one control step

    :param dxytheta: a control input
    :return:
    """
    dx_val = dxytheta[0]
    dy_val = dxytheta[1]
    dtheta = dxytheta[2]

    dx_dx_drift_ratio2 = self.dx_dx_drift_ratio * self.dx_dx_drift_ratio
    dy_dy_drift_ratio2 = self.dy_dy_drift_ratio * self.dy_dy_drift_ratio
    dist_angle_drift_ratio2 = self.dist_angle_drift_ratio * \
                              self.dist_angle_drift_ratio
    angle_dist_drift_ratio2 = self.angle_dist_drift_ratio * \
                              self.angle_dist_drift_ratio
    angle_angle_drift_ratio2 = self.angle_angle_drift_ratio * \
                               self.angle_angle_drift_ratio

    dx2 = dx_val * dx_val
    dy2 = dy_val * dy_val
    dtheta2 = dtheta * dtheta

    # cf. Probabilistic Robotics Sec. 5.4 alpha2 and alpha4
    covariance = np.identity(3)
    covariance[0, 0] = dx_dx_drift_ratio2 * dx2 + \
                       angle_dist_drift_ratio2 * dtheta2
    covariance[1, 1] = dy_dy_drift_ratio2 * dy2
    covariance[2, 2] = angle_angle_drift_ratio2 * dtheta2 + \
                       dist_angle_drift_ratio2 * dx2

    covariance[0, 2] = dx_dx_drift_ratio2 * 2 * dtheta * dx_val
    covariance[2, 0] = dx_dx_drift_ratio2 * 2 * dtheta * dx_val
    return covariance


  def compute_jacobians(self, state, one_u):
    """
    :param state: system state, x, y, theta
    :param one_u: control input
    :return:
    """
    f_mat = np.zeros([3, 3])
    w_mat = np.zeros([3, 3])
    delta = one_u.to_delta(self.velocity_y_threshold,
                           self.velocity_y_theta_ratio,
                           self.angular_rate_threshold)

    # partial derivative of x w.r.t. x
    f_mat[0, 0] = 1
    # partial derivative of x w.r.t. theta
    stheta = math.sin(state[2])
    ctheta = math.cos(state[2])
    f_mat[0, 2] = - stheta * delta[0] - ctheta * delta[1]

    # partial derivative of y w.r.t. y
    f_mat[1, 1] = 1
    # partial derivative of y w.r.t. theta
    f_mat[1, 2] = ctheta * delta[0] - stheta * delta[1]
    # partial derivative of theta w.r.t. theta
    f_mat[2, 2] = 1

    # w_mat = df/dw (Jacobian of state transition w.r.t. the noise)
    w_mat[0, 0] = ctheta
    w_mat[0, 1] = - stheta
    w_mat[1, 0] = stheta
    w_mat[1, 1] = ctheta
    w_mat[2, 2] = 1
    return f_mat, w_mat

  def get_odometry_covariance(self, timed_pose_list):
    """
    covariance over many steps for the odometry
    W and V must have (+/-)z axis along the gravity
    :param timed_pose_list: each row T_WV in [rostime tx ty tz qx qy qz qw]
    :return:
    """
    cov = np.zeros([3, 3])
    x_list = []
    u_list = []

    prev_time = 0.0
    prev_pose = np.identity(4)
    for index, timed_pose_vec in enumerate(timed_pose_list):
      curr_pose = tf_helpers.transformtransformation(timed_pose_vec[1:])
      if index > 0:
        delta_pose = np.dot(prev_pose, curr_pose)
        theta = tf_helpers.to_yaw_angle(delta_pose)
        xytheta = [delta_pose[0, 3], delta_pose[1, 3], theta]
        duration = timed_pose_vec[0] - prev_time
        u_list.append(Control(xytheta, duration.to_sec()))

      prev_time = timed_pose_vec[0]
      prev_pose = np.linalg.pinv(curr_pose)

      xytheta = timed_pose_vec[1:3] + \
                [tf_helpers.to_yaw_angle_quat(timed_pose_vec[4:])]
      x_list.append(xytheta)

    for index, one_u in enumerate(u_list):
      f_mat, w_mat = self.compute_jacobians(x_list[index], one_u)
      noise_cov = self.get_process_noise_covariance(one_u.rel_pose)
      cov = np.dot(f_mat, np.dot(cov, f_mat.transpose())) + \
          np.dot(w_mat, np.dot(noise_cov, w_mat.transpose()))
    return cov

  def get_delta_T_covariance(self, timed_pose_list): # pylint: disable=invalid-name
    """
    get the covariance of delta_T = T_Cp0Cp1_loc * (T_Cp0Cp1)^(-1)
        T_Cp0Cp1_loc = (T_GCp0)^(-1) * (T_GCp1)
        T_Cp0Cp1 = (T_WCp0)^(-1) * (T_WCp1)
    As all are expressed by x, y, theta, the cov(delta_T) is a 3x3 np array
    :param timed_pose_list:
    :return: cov(delta_T) a 3x3 np array
    """
    cov_loc = np.zeros([3, 3])
    cov_loc[0, 0] = self.loc_covariance_xy
    cov_loc[1, 1] = self.loc_covariance_xy
    cov_loc[2, 2] = self.loc_covariance_theta

    cov_odom = self.get_odometry_covariance(timed_pose_list)
    # if the ref traj is produced with Hokuyo lidar, its covariance should
    #  be smaller than that of wheel encoder. This does not matter much as
    #  cov_odom is typically rather small relative to loc. covariances
    return cov_odom + 2 * cov_loc
