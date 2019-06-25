#!/usr/bin/env python

'''
test functions of variance_localization_frequency
'''
from __future__ import print_function

import math
import os

import numpy as np
import rospy

import variance_localization_frequency as vlf
import tf_helpers

def test_stringtotime():
  """test stringtotime"""
  val = ["213789023321.", "419301.4132", "413880845.143801803487432",
         "41370470417084301", "4783740312778934"]
  exp_val = [rospy.Time(213789023321, 0),
             rospy.Time(419301, 413200000),
             rospy.Time(413880845, 143801803),
             rospy.Time(41370470, 417084301),
             rospy.Time(4783740312, 778934000)]
  res = []
  for index, entry in enumerate(val):
    if index <= 3:
      res.append(vlf.stringtotime(entry, 9))
    else:
      res.append(vlf.stringtotime(entry, 6))

  for index, entry in enumerate(res):
    assert exp_val[index] == entry


def test_load_time_trans_quat_txt():
  """test func"""
  file_path = os.path.abspath(__file__)

  src_dir = os.path.dirname(file_path)
  assert src_dir.endswith("test")
  dst_dir = os.path.join(src_dir, "data")

  pose_result_file = os.path.join(
      dst_dir, "temp_map_align_reloccheck_debug.txt")
  pose_result_dump = os.path.join(
      dst_dir, "dump_map_align_reloccheck_debug.txt")

  pose_reuslt = \
    ("1537164988072644864  25.8155 18.5445 1.00678   0.703727  0.459123 "
     "-0.357349  -0.40777    25.8054   19.1455 -0.274534  -0.547517 "
     "-0.471279  0.459518  0.516687  1.10769  0.202237  0\n"
     "1537164989271388928     24.5726    19.3661 -0.0944034   0.525554  "
     "0.499163 -0.501221 -0.472659    24.6623   19.1835 -0.277743  -0.518469 "
     "-0.502398  0.488537  0.490018  2.23132  0.344329  0\n"
     "1537164993067164928    21.1579   19.5594 -0.271317  -0.560544 -0.456284"
     "  0.456793  0.518589    21.1403   19.5371 -0.207932   -0.56298 -0.457661"
     "  0.452797  0.518241  1.47199  0.10874  1\n")
  with open(pose_result_file, "w") as stream:
    stream.write(pose_reuslt)

  timed_pose_list = vlf.load_time_trans_quat_txt(
      pose_result_file, 0, skiprows=0, decimaldigits=9)

  assert timed_pose_list[0][0] == rospy.Time(1537164988, 72644864)
  assert np.allclose(
      timed_pose_list[0][1:8],
      [25.8155, 18.5445, 1.00678, 0.703727, 0.459123, -0.357349, -0.40777])
  assert timed_pose_list[-1][0] == rospy.Time(1537164993, 67164928)
  assert np.allclose(
      timed_pose_list[-1][1:8],
      [21.1579, 19.5594, -0.271317, -0.560544, -0.456284, 0.456793, 0.518589])
  vlf.save_time_trans_quat_txt(timed_pose_list, pose_result_dump, 0)
  dumped_timed_pose_list = vlf.load_time_trans_quat_txt(pose_result_dump, 0)
  assert dumped_timed_pose_list == timed_pose_list

  os.remove(pose_result_file)
  os.remove(pose_result_dump)


def test_loc_metric_calculator():
  """test LocalizationMetricCalculator"""
  # localization result in TUM RGBD format
  T_GI_txt = "./test/data/2017-10-23_01-51-28__OfficeDemo-1023/loc_T_GI.txt"  # pylint: disable=invalid-name
  output_dir = os.path.dirname(T_GI_txt)
  mat_T_IC = np.array([[0.999908, -0.0130974, -0.00351362, 0.00681938],  # pylint: disable=invalid-name
                       [0.0130978, 0.999914, 7.85796e-05, 0.00237952],
                       [0.00351229, -0.000124593, 0.999994, 2.37739e-05],
                       [0, 0, 0, 1]])

  # localization result with the camera frame as the reference body frame
  # If T_IC is very close to Identity, this step may be extraneous
  chunk = vlf.load_time_trans_quat_txt(T_GI_txt, 0, delimiter=" ", skiprows=0)
  loc_T_GC = tf_helpers.right_multiply_transform(chunk, mat_T_IC) # pylint: disable=invalid-name

  T_GC_txt = os.path.join(output_dir, "temp_T_GC.txt")  # pylint: disable=invalid-name
  vlf.save_time_trans_quat_txt(loc_T_GC, T_GC_txt, line_format=0, delimiter=" ")

  # generated from tf_offline_processed.txt or pose_fisheye_tf.txt
  ref_T_Cp0_Cp_txt = ("./test/data/2017-10-23_01-51-28__OfficeDemo-1023/"  # pylint: disable=invalid-name
                      "pose_fisheye_tf.txt")
  chunk = vlf.load_time_trans_quat_txt(ref_T_Cp0_Cp_txt, 1)
  ref_T_WC = tf_helpers.right_multiply_transform( # pylint: disable=invalid-name
      chunk, vlf.CAMERA_T_PSEUDOCAMERA.transpose())
  ref_T_WC_txt = os.path.join(output_dir, "temp_ref_T_WC.txt") # pylint: disable=invalid-name
  vlf.save_time_trans_quat_txt(ref_T_WC, ref_T_WC_txt, 0)
  dist_interval = 5.0
  lmc = vlf.LocalizationMetricCalculator(T_GC_txt, ref_T_WC_txt, dist_interval)
  lmc.lc_gravity_deviation_tol = 8 * math.pi / 180
  lmc.calculate()
  lmc.print_msg()
  total_dist = 47.8413243151
  lmc.sanity_check(
      1.0, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      math.sqrt(13.0/500)*math.ceil(total_dist/dist_interval),
      total_dist)

  os.remove(T_GC_txt)
  os.remove(ref_T_WC_txt)


def test_compute_var_loc_freq():
  """test func"""
  # case 1 empty loc
  loc_time_list = None
  time_to_dists = [
      [1 * vlf.SECOND_TO_NANOSECOND, 0],
      [2 * vlf.SECOND_TO_NANOSECOND, 1],
      [3 * vlf.SECOND_TO_NANOSECOND, 2],
      [4 * vlf.SECOND_TO_NANOSECOND, 3],
      [5 * vlf.SECOND_TO_NANOSECOND, 4]
  ]
  dist_interval = 5
  var, std, loc_time_to_dist = vlf.compute_var_loc_freq(
      loc_time_list, time_to_dists, dist_interval)
  assert var == 0 and std == 0 and loc_time_to_dist is None

  # case 2 one loc, large dist_interval
  loc_time_list = [1.5 * vlf.SECOND_TO_NANOSECOND]
  var, std, loc_time_to_dist = vlf.compute_var_loc_freq(
      loc_time_list, time_to_dists, dist_interval)
  assert var == 0.0 and std == 0.0 and \
         loc_time_to_dist == [(1.5 * vlf.SECOND_TO_NANOSECOND, 0.5)]

  # case 2 two loc, medium dist_interval
  loc_time_list = [1.5 * vlf.SECOND_TO_NANOSECOND,
                   3.5 * vlf.SECOND_TO_NANOSECOND]
  dist_interval = 2.5
  var, std, loc_time_to_dist = vlf.compute_var_loc_freq(
      loc_time_list, time_to_dists, dist_interval)
  assert var == 0.0 and std == 0.0 and \
         loc_time_to_dist == [(1.5 * vlf.SECOND_TO_NANOSECOND, 0.5),
                              (3.5 * vlf.SECOND_TO_NANOSECOND, 2.5)]
  # case 3 three loc, medium dist interval
  loc_time_list = [1.5 * vlf.SECOND_TO_NANOSECOND,
                   3.5 * vlf.SECOND_TO_NANOSECOND,
                   4.5 * vlf.SECOND_TO_NANOSECOND]
  var, std, loc_time_to_dist = vlf.compute_var_loc_freq(
      loc_time_list, time_to_dists, dist_interval)
  assert np.allclose(var, 0.25) and np.allclose(std, 1.0/3) and \
         loc_time_to_dist == [(1.5 * vlf.SECOND_TO_NANOSECOND, 0.5),
                              (3.5 * vlf.SECOND_TO_NANOSECOND, 2.5),
                              (4.5 * vlf.SECOND_TO_NANOSECOND, 3.5)]

  # case 4 one outside loc, medium dist interval
  loc_time_list = [1.5 * vlf.SECOND_TO_NANOSECOND,
                   3.5 * vlf.SECOND_TO_NANOSECOND,
                   4.5 * vlf.SECOND_TO_NANOSECOND,
                   5.5 * vlf.SECOND_TO_NANOSECOND]
  var, std, loc_time_to_dist = vlf.compute_var_loc_freq(
      loc_time_list, time_to_dists, dist_interval)
  assert np.allclose(var, 0.25) and np.allclose(std, 1.0/3) and \
         loc_time_to_dist == [(1.5 * vlf.SECOND_TO_NANOSECOND, 0.5),
                              (3.5 * vlf.SECOND_TO_NANOSECOND, 2.5),
                              (4.5 * vlf.SECOND_TO_NANOSECOND, 3.5)]


def test_cumulative_distance():
  """test func"""
  points = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
  dimens = [0, 1, 2]

  # caes 1 None dimens
  dist_list = vlf.cumulative_distance(points, dimens=None)
  assert np.allclose(dist_list, np.array(
      [0.0, 3 * math.sqrt(3.0), 6 * math.sqrt(3.0)]))

  # caes 2 full dimens
  dist_list = vlf.cumulative_distance(points, dimens=dimens)
  assert np.allclose(dist_list, np.array(
      [0.0, 3 * math.sqrt(3.0), 6 * math.sqrt(3.0)]))

  # caes 3 partial dimens
  dist_list = vlf.cumulative_distance(points, dimens=[0, 1])
  exp_dist_list = np.array([0.0, 3 * math.sqrt(2.0), 6 * math.sqrt(2.0)])

  assert np.allclose(dist_list, exp_dist_list)
