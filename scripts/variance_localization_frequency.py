#!/usr/bin/env python

'''
compute the variance of localization frequency as defined in our ICRA19 paper
'''
from __future__ import print_function
import argparse

import math
import os
import sys
import warnings

import geometry_msgs
import numpy as np
import rospy
import tf

import differential_drive_model
import tf_helpers

MICRO_TO_NANO = 1000
NANOSECOND_TO_SECOND = 0.000000001
SECOND_TO_NANOSECOND = 1000000000
# the transform transforms a point in PSEUDOCAMERA frame to the CAMERA frame
CAMERA_T_PSEUDOCAMERA = np.array([[0, -1, 0, 0],
                                  [0, 0, -1, 0],
                                  [1, 0, 0, 0],
                                  [0, 0, 0, 1]])
# the approximate transform between the camera and IMU on a ZR300 sensor
APPROX_IMU_T_CAMERA = np.identity(4)

CHI_SQUARE_3DF_THRESHOLDS = [
    [0.90, 0.95, 0.975, 0.99, 0.999],
    [6.251, 7.815, 9.348, 11.345, 16.266]]

def compute_var_loc_freq(loc_time_list, time_to_dists, dist_interval):
  """
  compute the variance and standard deviation of localization frequency

  :param loc_time_list: list of timestamps of localized frames in nanosecond
  :param time_to_dists: timestamps in nanoseconds to traveled distances for
      a reference trajectory
      In practice, all locs should fall within the time range of the reference.
          The rare outside locs will be discounted.
  :param dist_interval: contiguous intervals along the traveled distance to
      compute localization frequencies
  :return:
      var_prf,
      std_prf,
      a list of tuples (localization time to traveled distance) for each loc
  """
  if loc_time_list is None or len(loc_time_list) < 1:
    warnings.warn("Warn: no loc is found and the variance is flat zero")
    return 0.0, 0.0, None

  time_range = (time_to_dists[0][0], time_to_dists[-1][0])
  loc_time_checklist = []
  outsider = 0
  for loc_time in loc_time_list:
    if loc_time >= time_range[0] and loc_time <= time_range[1]:
      loc_time_checklist.append(loc_time)
    else:
      outsider += 1
      warnings.warn("loc time {} ns falls outside of ({}, {}) ns".format(
          loc_time, time_range[0], time_range[1]))


  time_zero = time_to_dists[0][0]
  time_normalized_sec_list = []
  dist_list = []
  for row in time_to_dists:
    time_normalized_sec_list.append((row[0] - time_zero) * NANOSECOND_TO_SECOND)
    dist_list.append(row[1])
  query_time_sec_list = []
  for entry in loc_time_checklist:
    query_time_sec_list.append((entry - time_zero) * NANOSECOND_TO_SECOND)
  interp_dist_list = np.interp( # pylint: disable=no-member
      query_time_sec_list, time_normalized_sec_list, dist_list)

  total_dist = time_to_dists[-1][1]
  if total_dist < dist_interval:
    warnings.warn("Warn: the total traveled distance {} < the distance"
                  " interval {} for computing local localization frequency".
                  format(total_dist, dist_interval))

  num_bins = int(math.ceil(float(total_dist)/dist_interval))
  histogram_bins = np.zeros(num_bins)
  # bin boundaries are [0, d), [d, 2d), ... [(n-1)d, nd]

  for dist in interp_dist_list:
    bin_index = int(math.floor(dist / dist_interval))
    histogram_bins[bin_index] += 1

  num_locs = len(loc_time_list) - outsider
  var = np.var(histogram_bins)
  std = np.std(histogram_bins) * num_bins / num_locs
  return var, std, zip(loc_time_checklist, interp_dist_list)


def stringtotime(timestr, decimalcount=0):
  """
  convert a string to ros timestamp
  :param timestr:
  :param decimalcount: how many digits after the decimal dot are kept,
      only used when there is no decimal dot in the string
  :return: rospy.Time
  """
  if '.' in timestr:
    sec_frac = timestr.split('.')
    if sec_frac[1]:
      return rospy.Time(int(sec_frac[0]),
                        int(sec_frac[1][:9]) * 10**(max(9-len(sec_frac[1]), 0)))
    else:
      return rospy.Time(int(sec_frac[0]))
  else:
    if len(timestr) <= decimalcount:
      return rospy.Time(0, int(timestr) * 10**(9-decimalcount))
    else:
      return rospy.Time(int(timestr[0:-decimalcount]),
                        int(timestr[-decimalcount:]) * 10**(9-decimalcount))


def rostime_to_nanosecs(rostime):
  """
  :param rostime:
  :return: a large int of nano seconds
  """
  return rostime.nsecs + rostime.secs * SECOND_TO_NANOSECOND


def load_time_trans_quat_txt(tffile, line_format, delimiter=" ",
                             skiprows=0, decimaldigits=6):
  """
  load a txt file

  :param tffile:
  :param line_format: if 0 TUM format t[s] x y z qx qy qz qw
      if 1 internal format t[us] x y z qw qx qy qz
      Because this function does not do normalization,
          for better precision, qx qy qz qw should have 9 decimal digits

  :return: a list of timed poses, each entry is
      [rospy time, tx, ty, tz, qx, qy, qz, qw]
  """
  if line_format != 0 and line_format != 1:
    raise NotImplementedError(
        "line_format {} not implemented yet".format(line_format))
  chunk = []
  with open(tffile, "r") as stream:
    for index, line in enumerate(stream):
      if index < skiprows:
        continue
      if delimiter == " ":
        time_pose_str = line.split()
      else:
        time_pose_str = line.split(delimiter)
      timestamp = stringtotime(time_pose_str[0], decimaldigits)
      row = [timestamp] + [float(x) for x in time_pose_str[1:]]
      if line_format == 1:
        w_val = row[4]
        row[4:7] = row[5:8]
        row[7] = w_val
      chunk.append(row)
  return chunk


def save_time_trans_quat_txt(rostime_txyz_qxyzw_list, tffile,
                             line_format=0, delimiter=" "):
  """
  save a list of entries to a file

  :param rostime_txyz_qxyzw_list: each row
      [rostime, tx, ty, tz, qx, qy, qz, qw]
  :param line_format: if 0 TUM format t[s] x y z qx qy qz qw
      if 1 internal format t[us] x y z qw qx qy qz
  :param delimiter:
  :return:
  """
  if line_format != 0 and line_format != 1:
    raise NotImplementedError(
        "line_format {} not implemented yet".format(line_format))
  with open(tffile, "w") as stream:
    for row in rostime_txyz_qxyzw_list:
      timestr = '.'.join([str(row[0].secs), "{0:09d}".format(row[0].nsecs)])
      floats_str = []
      for index, val in enumerate(row[1:]):
        if index < 2:
          floats_str.append("{:.6f}".format(val))
        else:
          floats_str.append("{:.10f}".format(val))

      if line_format == 1:
        w_str = floats_str[6]
        floats_str[4:7] = floats_str[3:6]
        floats_str[3] = w_str
      stream.write(delimiter.join([timestr] + floats_str))
      stream.write("\n")


def create_transformer(rostime_txyz_qxyzw_list, target_frame, source_frame):
  """
    create a ros transformer from a txt file contains poses W_T_B

    :param rostime_txyz_qxyzw_list: each row
        [rostime, tx, ty, tz, qx, qy, qz, qw]
    :param target_frame: label of the world frame {W}
    :param source_frame: label of the reference body frame {B}
    :return: the transformer that can be used for querying interpolated poses
    """
  maxduration = rospy.Duration(3600.0)
  transformer = tf.Transformer(True, maxduration)

  for row in rostime_txyz_qxyzw_list:
    monk = geometry_msgs.msg.TransformStamped()
    monk.header.stamp = row[0]
    monk.header.frame_id = target_frame
    monk.child_frame_id = source_frame
    monk.transform.translation.x = row[1]
    monk.transform.translation.y = row[2]
    monk.transform.translation.z = row[3]
    monk.transform.rotation.x = row[4]
    monk.transform.rotation.y = row[5]
    monk.transform.rotation.z = row[6]
    monk.transform.rotation.w = row[7]
    transformer.setTransform(monk)
  return transformer


def create_transformer_from_txt(
    tffile, target_frame, source_frame, line_format=0):
  """
  create a ros transformer from a txt file contains poses W_T_B
  :param target_frame: label of the world frame {W}
  :param source_frame: label of the reference body frame {B}
  :param line_format: if 0 TUM format t[s] x y z qx qy qz qw
      if 1 internal format t[us] x y z qw qx qy qz
  :return: the transformer that can be used for querying interpolated poses
  """
  if not os.path.isfile(tffile):
    raise OSError(2, 'No such tf file', tffile)
  chunk = load_time_trans_quat_txt(tffile, line_format)
  return create_transformer(chunk, target_frame, source_frame)


def query_transformation(transformer, timelist, target_frame, source_frame):
  """
  interpolate transformations at a list of epochs for a reference trajectory
      stored in transformer
  :param transformer: ROS structure holding the reference trajectory
  :param timelist: list of ROS times
  :return: list of interpolated transforms at the specified epochs,
      Each transform is [tx ty tz qx qy qz qw]
      None is added to the list if the transform at an epoch is not available
  """
  transformlist = []
  for stamp in timelist:
    if transformer.canTransform(target_frame, source_frame, stamp):
      transform = transformer.lookupTransform(
          target_frame, source_frame, stamp)
      mytransform = transform[0]
      mytransform.extend(transform[1])
      transformlist.append(mytransform)
    else:
      transformlist.append(None)
  return transformlist

def crop_timed_pose_list(
    begin_timed_pose, end_timed_pose, all_timed_poses):
  """

  :param begin_timed_pose: [rostime, [tx ty tz qx qy qz qw]]
  :param end_timed_pose: [rostime, [tx ty tz qx qy qz qw]]
  :param all_timed_poses: a list of [rostime, [tx ty tz qx qy qz qw]]
  :return: the segment of all_time_poses between bein_timed_pose and
      end_timed_pose, and if need be, including the begin_timed_pose or
      end_timed_pose
  """
  if (begin_timed_pose[0] < all_timed_poses[0][0]) or \
      (end_timed_pose[0] > all_timed_poses[-1][0]):
    raise ValueError("Cropping outside the boundary!")

  cropped_timed_poses = []
  begin_index = -1
  begin_append = True
  end_index = -1
  end_append = True
  for index, timed_pose in enumerate(all_timed_poses):
    if timed_pose[0] < begin_timed_pose[0]:
      begin_index = index
    elif timed_pose[0] == begin_timed_pose[0]:
      begin_index = index
      begin_append = False

    if timed_pose[0] < end_timed_pose[0]:
      end_index = index
    elif timed_pose[0] == end_timed_pose[0]:
      end_index = index
      end_append = False
    else:
      break
  if begin_append:
    begin_index += 1
    cropped_timed_poses.append(begin_timed_pose)

  for index in range(begin_index, end_index + 1):
    cropped_timed_poses.append(all_timed_poses[index])

  if end_append:
    cropped_timed_poses.append(end_timed_pose)
  return cropped_timed_poses


def detect_loc_outliers(loctimes, loc_T_GCp, ref_T_WCp, # pylint: disable=invalid-name
                        ref_timed_T_WCp_full, # pylint: disable=invalid-name
                        mahal_tol, lc_gravity_deviation_tol):
  """
  detect outliers in localizations relative to a reference trajectory
  :param loctimes: rospy epochs of localizations
  :param loc_T_GCp: poses of localizations at every epoch with the Cp
      frame being the reference body frame.
      The reference body frame should have its z axis roughly along
          the gravity direction
      Each pose [tx ty tz qx qy qz qw]
  :param ref_T_WCp: poses of reference trajectory at every epoch have the
      same length and the same reference body frame as loc_T_GCp
      Each pose [tx ty tz qx qy qz qw]
      ref_T_WCp may have None entries as some poses may not be interpolatable
  :return: estimated number of outliers,
      and the probabilities of a loc being an outlier
      if the prob is 0.5, it means that the loc has been adjacent to an outlier
      if the prob is 1, it means that the loc is probably an outlier,
  """
  outliercount = 0
  outlierprob = [0.0] * len(loc_T_GCp)
  if len(loc_T_GCp) < 2:
    return outliercount, outlierprob
  if len(loctimes) != len(loc_T_GCp):
    raise ValueError("Should provide the same number of epochs and "
                     "localizations, which are {} and {}, resp.".format(
                         len(loctimes), len(loc_T_GCp)))

  if len(loc_T_GCp) != len(ref_T_WCp):
    raise ValueError("Should provide localizations and reference poses at "
                     "the same epochs. Current #loc. {} #ref_poses {}".format(
                         len(loc_T_GCp), len(ref_T_WCp)))

  reftimeandpose = None # the first of a pair of localizations

  for index, temp_T_GCp in enumerate(loc_T_GCp):  # pylint: disable=invalid-name
    if not ref_T_WCp[index]:
      print("Reference pose not available at {} sec".
            format(loctimes[index].to_sec()))
      continue
    # gravity consistency check
    T_GCp1 = tf_helpers.transformtransformation(temp_T_GCp)  # pylint: disable=invalid-name
    T_WCp1 = tf_helpers.transformtransformation(ref_T_WCp[index])  # pylint: disable=invalid-name

    T_WG = np.dot(T_WCp1, np.linalg.pinv(T_GCp1))  # pylint: disable=invalid-name
    gravity_consistent = math.acos(T_WG[2, 2]) < lc_gravity_deviation_tol

    if not gravity_consistent:
      print("failed gravity check at {} sec".format(loctimes[index].to_sec()))
      outlierprob[index] += 1.0
      outliercount += 2
      continue

    if not reftimeandpose: # first entry
      reftimeandpose = [loctimes[index], index]
      continue

    T_GCp0 = tf_helpers.transformtransformation(loc_T_GCp[reftimeandpose[1]])  # pylint: disable=invalid-name
    T_WCp0 = tf_helpers.transformtransformation(ref_T_WCp[reftimeandpose[1]])  # pylint: disable=invalid-name

    T_Cp0W = np.linalg.pinv(T_WCp0)  # pylint: disable=invalid-name
    T_Cp0Cp1 = np.dot(T_Cp0W, T_WCp1)  # pylint: disable=invalid-name

    # T_GCp0 may have a very large translation,
    # causing its pinv(3,3) close to zero
    T_Cp0Cp1_loc = np.dot(np.linalg.pinv(T_GCp0), T_GCp1)  # pylint: disable=invalid-name
    deltaT = np.dot(T_Cp0Cp1_loc, np.linalg.pinv(T_Cp0Cp1))  # pylint: disable=invalid-name

    delta_vec = tf_helpers.to_xy_theta(deltaT)
    delta_arr = np.array(delta_vec)

    ddm = differential_drive_model.DifferentialDriveModel(
        0.03, 0.01, 0.01, 0.0076)
    try:
      timed_pose_list = crop_timed_pose_list(
          [reftimeandpose[0]] + ref_T_WCp[reftimeandpose[1]],
          [loctimes[index]] + ref_T_WCp[index],
          ref_timed_T_WCp_full)
    except ValueError as err:
      print("crop begin {} data begin {} crop end {} data end {}".format(
          reftimeandpose[0].to_sec(), ref_timed_T_WCp_full[0][0].to_sec(),
          loctimes[index].to_sec(), ref_timed_T_WCp_full[-1][0].to_sec()
      ))
      raise err

    delta_cov = ddm.get_delta_T_covariance(timed_pose_list)

    dist = math.sqrt(np.dot(
        delta_arr, np.dot(np.linalg.pinv(delta_cov), delta_arr.transpose())))

    if dist > mahal_tol:
      outlierprob[reftimeandpose[1]] += 0.5
      outlierprob[index] += 0.5
      outliercount += 1

    print("dist {} mahal_tol {} for {} and {}".format(
        dist, mahal_tol, reftimeandpose[0].to_sec(), loctimes[index].to_sec()))

    # print("T_Cp0Cp1\n{}\ndeltaT\n{}\ndelta_cov\n{}\ndist {} mahal {}".format(
    #     T_Cp0Cp1, deltaT, delta_cov,
    #     dist, mahal_tol))

    reftimeandpose[0] = loctimes[index]
    reftimeandpose[1] = index

  # unfortunately, an inlier sandwiched by two outliers will have a prob 1.0
  # the below trick tries to alleviate the issue
  for index, val in enumerate(outlierprob):
    if index > 1:
      if outlierprob[index - 2] == 1.0 and outlierprob[index - 1] == 1.0 \
          and val == 1.0:
        outlierprob[index - 1] = 0.0

  # sanity check
  count = 0
  for val in outlierprob:
    if val >= 1.0:
      count += 1
  outliercount = outliercount / 2.0
  tolerance = 2.0
  if abs(outliercount - count) >= tolerance:
    warnings.warn("#outlier computed from outlier probabilities {} deviates"
                  " more than {} from the estimated #outlier {}".format(
                      count, tolerance, outliercount))
  print("outlier count {} and prob {}".format(outliercount, outlierprob))
  return outliercount, outlierprob


def cumulative_distance(points, dimens=None):
  """
  compute the traveled distance given a sequence of way points

  :param points: n x dimen numpy array where n is the number of points
  :param dimens: list of indices of interested dimensions of points (0-based)
  :return: cummulative distances, 0 for the first point
  """

  if dimens is None:
    distance_on_dim = points[1:, :] - points[:-1, :]
  else:
    distance_on_dim = points[1:, dimens] - points[:-1, dimens]
  distance_squared = distance_on_dim ** 2
  distance_sum = np.sum(distance_squared, axis=1)
  distance = np.sqrt(distance_sum) # pylint: disable=no-member
  distance = np.insert(distance, 0, 0.0, axis=0)  # pylint: disable=no-member
  return np.cumsum(distance)


class LocalizationMetricCalculator(object):
  """
  compute the localization metrics given localization results and
      reference odometry with the camera frame as the body frame
  The localization metrics including number of outliers,
      normalized number of true positives by the traveled distance,
      the variance of distributions of localizations
      see the paper for more details
  """
  def __init__(self, loc_result_txt, ref_traj_txt,
               dist_interval):
    """
    Both loc_result_txt and ref_traj_txt are in TUM RGBD format, i.e.,
        each line: time[sec] tx[m] ty[m] tz[m] qx qy qz qw
    Poses of the two files should refer to the same reference body frame,
        typically, the camera frame
    """

    self.loc_T_GC_txt = loc_result_txt  # pylint: disable=invalid-name
    self.ref_T_WC_txt = ref_traj_txt  # pylint: disable=invalid-name

    self.mahal_tol = CHI_SQUARE_3DF_THRESHOLDS[1][2]
    self.dist_interval = dist_interval
    self.on_all_positives = False  # variance on all positives or true positives
    # relax it because Cp0 may not be so aligned to the gravity
    self.lc_gravity_deviation_tol = 15*math.pi/180

    # G and W frame are world frames with (+/-)z along the gravity,
    # they are not necessarily the same
    self.loc_timed_T_GCp = []  # pylint: disable=invalid-name
    self.ref_timed_T_WCp = []  # pylint: disable=invalid-name
    self.target_frame = "WORLD"
    self.source_frame = "PSEUDOCAMERA"

    self.outlier_count = 0
    self.outlier_prob = []

    self.var_prf = 0.0 # var of #loc. in bins
    # each bin is a distance interval on the trajectory
    self.std_prf = 0.0 # std of #loc. in bins divided by the mean
    # list of [loc_time_ns, loc_dist in meters from the session start]
    self.loc_time_to_dist = []
    self.total_dist = 0.0
    self.total_time = 0.0

  def calculate(self):
    """compute the localization metrics"""
    create_transformer_from_txt(
        self.loc_T_GC_txt, self.target_frame, self.source_frame, 1)
    loc_timed_T_GC = load_time_trans_quat_txt(self.loc_T_GC_txt, 0)  # pylint: disable=invalid-name
    ref_timed_T_WC = load_time_trans_quat_txt(self.ref_T_WC_txt, 0)  # pylint: disable=invalid-name
    timespan = ref_timed_T_WC[-1][0] - ref_timed_T_WC[0][0]
    self.total_time = timespan.to_sec()

    # we rotate the camera frame so that its z axis pointing close to the
    # negative gravity direction, aka, pseudo camera {Cp} frame, in order
    #  to properly compute the covariance of the yaw angle
    self.loc_timed_T_GCp = tf_helpers.right_multiply_transform(
        loc_timed_T_GC, CAMERA_T_PSEUDOCAMERA)
    self.ref_timed_T_WCp = tf_helpers.right_multiply_transform(
        ref_timed_T_WC, CAMERA_T_PSEUDOCAMERA)
    ref_T_WCp_transformer = create_transformer(  # pylint: disable=invalid-name
        self.ref_timed_T_WCp, self.target_frame, self.source_frame)
    timelist = [x[0] for x in self.loc_timed_T_GCp]
    ref_T_WCp_sample = query_transformation(  # pylint: disable=invalid-name
        ref_T_WCp_transformer, timelist, self.target_frame, self.source_frame)

    loc_T_GCp = [x[1:] for x in self.loc_timed_T_GCp] # pylint: disable=invalid-name
    try:
      self.outlier_count, self.outlier_prob = detect_loc_outliers(
          timelist, loc_T_GCp, ref_T_WCp_sample, self.ref_timed_T_WCp,
          self.mahal_tol, self.lc_gravity_deviation_tol)
    except ValueError as err:
      print("loc_T_GC_txt {}\nref_T_WC_txt {}".format(
          self.loc_T_GC_txt, self.ref_T_WC_txt))
      raise err

    loc_time_ns = []

    # discard false positives
    if not self.on_all_positives:
      for index, rostime in enumerate(timelist):
        if self.outlier_prob[index] < 1.0:
          loc_time_ns.append(rostime_to_nanosecs(rostime))
    else:
      loc_time_ns = [rostime_to_nanosecs(rostime)
                     for rostime in timelist]

    ref_waypoints = []
    for timed_pose in self.ref_timed_T_WCp:
      ref_waypoints.append(timed_pose[1:4])

    tf_dists = cumulative_distance(np.array(ref_waypoints))

    time_to_dists = []
    for index, entry in enumerate(self.ref_timed_T_WCp):
      time_to_dists.append(
          [rostime_to_nanosecs(entry[0]),
           tf_dists[index]])
    self.total_dist = tf_dists[-1]
    self.var_prf, self.std_prf, self.loc_time_to_dist = compute_var_loc_freq(
        loc_time_ns, time_to_dists, self.dist_interval)

  def print_msg(self):
    """
    print the computed localization stats
    :return:
    """

    print("outlier prob for each loc {}".format(self.outlier_prob))
    msg = r"#loc. {} #outlier {}".format(len(self.outlier_prob),
                                         self.outlier_count)

    std_str = "{:.3f}".format(self.std_prf)
    dist_str = "{:.3f}".format(self.total_dist)
    dura_str = "{:.3f}".format(self.total_time)
    interval_str = "{:.1f}".format(self.dist_interval)
    msg += r" $s_{PRF}$ " + std_str + \
           " with L " + dist_str + r" m timespan " + dura_str + \
           r" secs and $\Delta_l$ "+ interval_str + " m"
    print(msg)
    return msg

  def sanity_check(
      self, outlier_count=None, outlier_prob=None,
      std_prf=None, total_dist=None):
    """
    check if the results are ok
    :param outlier_count:
    :param outlier_prob:
    :param var_prf:
    :return:
    """
    if outlier_count is not None:
      assert outlier_count == self.outlier_count
    if outlier_prob is not None:
      assert np.allclose(outlier_prob, self.outlier_prob)
    if std_prf is not None:
      assert np.allclose(std_prf, self.std_prf)
    if total_dist is not None:
      assert np.allclose(total_dist, self.total_dist)


def parse_args():
  """parse arguments"""
  parser = argparse.ArgumentParser(
      description='Detect outliers and compute variance of localization '
                  'frequencies over fixed traveled distances. The reference '
                  'trajectory can be obtained from high-fidelity lidar mapping'
                  ' or low-fidelity wheel odometry')

  parser.add_argument('tf_txt', metavar='tf_txt', nargs='?',
                      help='The estimated poses for the camera frame, {T_WC}.'
                           'Each line: time[sec] tx ty tz qx qy qz qw',
                      required=True)

  parser.add_argument('loc_txt', metavar='loc_txt', nargs='?',
                      help='The localizations w.r.t the camera frame, T_{GC}.'
                           'Each line: time[sec] tx ty tz qx qy qz qw',
                      required=True)

  parser.add_argument("--dist_interval", type=float, nargs='?',
                      default=5.0,
                      help='Time interval to accumulate local localizations'
                           ' (default: %(default)s)', required=False)

  if len(sys.argv) < 2:
    parser.print_help()
    sys.exit(1)
  parsed = parser.parse_args()
  return parsed


def main():
  '''main body'''
  parsed = parse_args()
  lmc = LocalizationMetricCalculator(
      parsed.loc_txt, parsed.tf_txt, parsed.dist_interval)
  lmc.calculate()
  lmc.print_msg()

if __name__ == "__main__":
  main()
