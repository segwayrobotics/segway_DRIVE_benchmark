#!/usr/bin/env python
"""
draw fused poses, localizations onto the raster lidar map
"""

from __future__ import print_function

import json
import math
import os
import sys

import yaml
from enum import Enum

import blockbar
# the below is behind blockbar because there is matplotlib.use('Agg')
import matplotlib.pyplot as plt # pylint: disable=wrong-import-order
from matplotlib.backends.backend_pdf import PdfPages # pylint: disable=wrong-import-order

import numpy as np
import rospy


import tf_helpers
import folder_helper
import variance_localization_frequency as vlf


def read_time_pose_from_json(filename):
  """read pose from a json file"""
  with open(filename, 'r') as load_f:
    load_dict = json.load(load_f)
    xval = float(load_dict['translation']['x'])
    yval = float(load_dict['translation']['y'])
    zval = float(load_dict['translation']['z'])
    q_x = float(load_dict['rotation']['i'])
    q_y = float(load_dict['rotation']['j'])
    q_z = float(load_dict['rotation']['k'])
    q_w = float(load_dict['rotation']['w'])
    pose = [xval, yval, zval, q_x, q_y, q_z, q_w]
    time = rospy.Duration(float(load_dict["delay"]))
    return time, pose


def compute_real_coords_x(ticks, resolution, offset):
  """
  convert a pixel coord to the real coord in x
  :param ticks:
  :param resolution:
  :param offset:
  :return:
  """
  labels = []
  for tick in ticks:
    face_val = tick * resolution + offset
    labels.append("{:.2f}".format(face_val))
  return labels


def compute_real_coords_y(ticks, resolution, offset, raster_rows):
  """
  convert a pixel coord to the real coord in y
  :param ticks:
  :param resolution:
  :param offset:
  :return:
  """
  labels = []
  for tick in ticks:
    face_val = (raster_rows - tick) * resolution + offset
    labels.append("{:.2f}".format(face_val))
  return labels


def compute_raster_coords(xval, yval, originx, originy,
                          raster_rows, resolution):
  """
  convert real coords to raster coords
  :param xval:
  :param yval:
  :param originx:
  :param originy:
  :param raster_rows:
  :param resolution:
  :return:
  """
  return [(xval - originx) / resolution,
          raster_rows - (yval - originy) / resolution]


def squash_poses_onto_raster(timed_poses, map_info, raster_rows):
  """
  convert 3d poses to 2d pixel coords
  :param timed_poses: [rostime x y z qx qy qz qw]
  :param map_info: the origin and resolution of the map
  :param raster_rows: rows of the raster map
  :return:
  """
  pose_x = []
  pose_y = []
  for timed_pose in timed_poses:
    xval = (timed_pose[1] - map_info["origin"][0]) / map_info["resolution"]
    yval = raster_rows - \
        (timed_pose[2] - map_info["origin"][1]) / map_info["resolution"]
    pose_x.append(int(xval))
    pose_y.append(int(yval))
  return pose_x, pose_y


def load_and_convert_poses(pose_result_file, skip_rows=0,
                           left_multiplier=None, right_multiplier=None,
                           decimaldigits=9, delay_time=None):
  """
  load poses and apply transforms if necessary
  :param pose_result_file:
  :param skip_rows:
  :param left_multiplier:
  :param right_multiplier:
  :return:
  """
  timed_pose_list = vlf.load_time_trans_quat_txt(
      pose_result_file, 0, skiprows=skip_rows, decimaldigits=decimaldigits)
  if left_multiplier is not None:
    timed_pose_list = \
        tf_helpers.left_multiply_transform(timed_pose_list, left_multiplier)
  if right_multiplier is not None:
    timed_pose_list = \
        tf_helpers.right_multiply_transform(timed_pose_list, right_multiplier)
  if delay_time is not None:
    for index, value in enumerate(timed_pose_list):
      timed_pose_list[index][0] = value[0] + delay_time
  return timed_pose_list


def plot_traj_on_raster_map(map_pgm, map_yaml, timed_linked_poses,
                            timed_marker_poses, title_str, output_dir):
  """

  :param map_pgm:
  :param map_yaml:
  :param timed_linked_poses: poses of the trajectory expressed in the world
      of the lidar map
  :param timed_marker_poses: poses of the localizations
  :param title_str:
  :param output_dir:
  :return:
  """
  map_info = yaml.safe_load(open(map_yaml))
  img = plt.imread(map_pgm)
  size_y = img.shape[0]
  size_x = img.shape[1]

  pose_x, pose_y = squash_poses_onto_raster(
      timed_linked_poses, map_info, size_y)
  print("pose_x size {}".format(len(pose_x)))
  marker_x = None
  marker_y = None
  if timed_marker_poses:
    marker_x, marker_y = squash_poses_onto_raster(
        timed_marker_poses, map_info, size_y)
    print("marker_x size {}".format(len(marker_x)))

  fig, ax = plt.subplots() # pylint: disable=invalid-name
  ax.imshow(img, plt.cm.gray) # pylint: disable=no-member

  # ax.autoscale(False) # unnecessary

  ax.plot(pose_x, pose_y, "-k", label="traj.")
  if marker_x is not None and marker_y is not None:
    ax.plot(marker_x, marker_y, "g+", label="loc.")

  ax.plot(pose_x[0], pose_y[0], "or", label="begin")
  ax.plot(pose_x[-1], pose_y[-1], "sr", label="end")

  ax.set_xbound(lower=0.0, upper=size_x)
  ax.set_ybound(lower=0.0, upper=size_y)

  xticks = ax.get_xticks()
  xlabels = compute_real_coords_x(
      xticks, map_info["resolution"], map_info["origin"][0])

  yticks = ax.get_yticks()
  ylabels = compute_real_coords_y(
      yticks, map_info["resolution"], map_info["origin"][1], size_y)

  ax.set_xticklabels(xlabels)
  ax.set_yticklabels(ylabels)

  plt.grid(True)
  plt.title(title_str, fontsize='x-small')
  plt.legend(loc=0, fontsize="xx-small")

  pdfname = os.path.join(output_dir, "overlay.pdf")
  pdf = PdfPages(pdfname)
  pdf.savefig(fig, bbox_inches='tight')
  pdf.close()

  pngname = os.path.join(output_dir, "overlay.png")
  plt.savefig(pngname)
  plt.close('all')
  print("output files named overlay are under {}".format(output_dir))
  return fig


def draw_loc_histogram(std_prf, time_to_dist, dist_interval, output_dir):
  """
  draw the histogram of loc dists
  :param time_to_dist:
  :return:
  """
  dists = [time_dist[1] for time_dist in time_to_dist]
  max_dist = max(dists)
  num_bins = int(math.ceil(float(max_dist)/dist_interval))
  bin_edges = [index * dist_interval for index in range(num_bins + 1)]

  fig, ax = plt.subplots()  # pylint: disable=invalid-name
  ax.hist(dists, bin_edges, facecolor='g', alpha=0.75)

  plt.xlabel('Distance from the start[m]')
  plt.ylabel('#loc.')
  var_str = "{:.3f}".format(std_prf)
  plt.title(r'Histogram of localizations with $s_{PRF}$ ' + var_str)
  plt.grid(True)

  pdfname = os.path.join(output_dir, "loc_hist.pdf")
  pdf = PdfPages(pdfname)
  pdf.savefig(fig, bbox_inches='tight')
  pdf.close()

  pngname = os.path.join(output_dir, "loc_hist.png")
  plt.savefig(pngname)
  plt.close('all')
  # plt.show()
  return fig


class TrajResultType(Enum):
  """
  the traj type from different localization modules, eg., amcl pose result and
      map_align_pose_debug
  """
  UNKNOWN = 0
  AMCL_LOC = 1 # amcl lidar localization
  APR_LOC = 2 # appearance based place recognition


def draw_traj_on_map_one_dir(map_folder, pose_result_dir, calib_dir):
  """

  :param map_folder:  containing map.pgm and yaml
  :param pose_result_dir: dir containing candidate traj pose txt
  :param calib_dir: dir containing calibration json files
  :return: a handle to fig
  """
  map_pgm = os.path.join(map_folder, "map.pgm")
  map_yaml = os.path.join(map_folder, "map.yaml")
  skip_rows = 0
  left_multi_mat = None
  right_multi_mat = None
  trt = TrajResultType.UNKNOWN
  candidates = ["amcl_pose_result.txt", "map_align_pose_debug.txt"]

  if os.path.isfile(os.path.join(pose_result_dir, candidates[0])):
    trt = TrajResultType.AMCL_LOC
  elif os.path.isfile(os.path.join(pose_result_dir, candidates[1])):
    trt = TrajResultType.APR_LOC
  else:
    raise NotImplementedError(
        "Missing known pose files under {}".format(pose_result_dir))

  calib_json = ["lidarworld_T_vioworld.json", "calibration_optimized.json"]
  loc_basename = "map_align_reloccheck_debug.txt"

  B_T_I_json = None # pylint: disable=invalid-name
  if trt == TrajResultType.APR_LOC:
    pose_result_file = os.path.join(pose_result_dir, candidates[1])
    skip_rows = 1

    Wl_T_M_json = os.path.join(calib_dir, calib_json[0]) # pylint: disable=invalid-name
    B_T_I_json = os.path.join(calib_dir, calib_json[1]) # pylint: disable=invalid-name
    if os.path.isfile(Wl_T_M_json):
      _, left_multi_vec = read_time_pose_from_json(Wl_T_M_json)
      left_multi_mat = tf_helpers.transformtransformation(left_multi_vec)
    if os.path.isdir(B_T_I_json):
      _, right_multi_vec = read_time_pose_from_json(B_T_I_json)
      right_multi_mat = tf_helpers.transformtransformation(right_multi_vec)
      right_multi_mat = np.linalg.pinv(right_multi_mat)

  elif trt == TrajResultType.AMCL_LOC:
    pose_result_file = os.path.join(pose_result_dir, candidates[0])
    skip_rows = 0

  timed_traj_poses = load_and_convert_poses(
      pose_result_file, skip_rows=skip_rows,
      left_multiplier=left_multi_mat, right_multiplier=right_multi_mat)

  timed_marker_poses = None
  msg = ""
  loc_time_to_dist = None
  if trt == TrajResultType.APR_LOC:
    loc_result_file = os.path.join(pose_result_dir, loc_basename)
    if os.path.isfile(loc_result_file):
      timed_marker_poses = load_and_convert_poses(
          loc_result_file, skip_rows=1,
          left_multiplier=left_multi_mat, right_multiplier=right_multi_mat)

      info_debug_txt = os.path.join(pose_result_dir, "info_debug.txt")
      test_data_dir = ""
      with open(info_debug_txt, "r") as stream:
        for index, line in enumerate(stream):
          if index == 0:
            str_list = line.split(":")
            test_data_dir = str_list[1].strip()
            break
      print("test data dir {} vs pose_result_file path {}".format(
          test_data_dir, os.path.basename(os.path.dirname(pose_result_file))))
      assert os.path.basename(test_data_dir) in pose_result_file
      # hack begin
      fake_test_data_dir = "/persist/data/huyue_lidar_dataset/B6-B1/lidar"
      if "B2" in pose_result_file:
        fake_test_data_dir = "/persist/data/huyue_lidar_dataset/B2-F1/lidar"
      test_data_dir = os.path.join(
          fake_test_data_dir, os.path.basename(test_data_dir))
      # hack end
      ref_T_W_Cp_txt = folder_helper.find_file_under_folder( # pylint: disable=invalid-name
          test_data_dir, "pose_fisheye_tf.txt")
      msg, std_prf, loc_time_to_dist, dist_interval = \
          compute_loc_metrics_odom(loc_result_file, ref_T_W_Cp_txt)

  test_data_basename = os.path.basename(os.path.dirname(pose_result_file))
  # hack begin
  # test_data_basename = test_data_basename.split("_2019-05")[0]
  # hack end

  title_str = test_data_basename + "\n" + msg
  figs = []
  if timed_traj_poses:
    figs.append(plot_traj_on_raster_map(
        map_pgm, map_yaml, timed_traj_poses, timed_marker_poses,
        title_str, pose_result_dir))
    if loc_time_to_dist is not None:
      figs.append(draw_loc_histogram(std_prf, loc_time_to_dist,
                                     dist_interval, pose_result_dir))
  return figs, title_str


def compute_loc_metrics_hokuyo(T_GI_txt, ref_T_Wl_L, T_LI_json): # pylint: disable=invalid-name
  """

  :param T_GI_txt: map_align_reloccheck_debug.txt
  :param ref_T_Wl_L: hokuyo lidar tf_offline_processed.txt
  :param T_LI_json: calibration_optimized.json for hokuyo lidar,
      tied to ref_T_Wl_L
  :return:
  """
  output_dir = os.path.dirname(T_GI_txt)
  mat_T_IC = np.array([[0.999908, -0.0130974, -0.00351362, 0.00681938],  # pylint: disable=invalid-name
                       [0.0130978, 0.999914, 7.85796e-05, 0.00237952],
                       [0.00351229, -0.000124593, 0.999994, 2.37739e-05],
                       [0, 0, 0, 1]])

  # localization result with the camera frame as the reference body frame
  # If T_IC is very close to Identity, this step may be extraneous

  timed_marker_poses = load_and_convert_poses(
      T_GI_txt, skip_rows=1,
      left_multiplier=None, right_multiplier=mat_T_IC)

  T_GC_txt = os.path.join(output_dir, "temp_T_GC.txt")  # pylint: disable=invalid-name
  vlf.save_time_trans_quat_txt(timed_marker_poses, T_GC_txt,
                               line_format=0, delimiter=" ")

  delay, right_multi_vec = read_time_pose_from_json(T_LI_json)
  right_multi_mat = tf_helpers.transformtransformation(right_multi_vec)
  right_multi_mat = np.dot(right_multi_mat, mat_T_IC)

  timed_traj_poses = load_and_convert_poses(
      ref_T_Wl_L, skip_rows=1,
      left_multiplier=None, right_multiplier=right_multi_mat, delay_time=delay)

  ref_T_WC_txt = os.path.join(output_dir, "temp_ref_T_WC.txt") # pylint: disable=invalid-name
  vlf.save_time_trans_quat_txt(timed_traj_poses, ref_T_WC_txt, 0)

  lmc = vlf.LocalizationMetricCalculator(T_GC_txt, ref_T_WC_txt, 5.0)
  lmc.calculate()
  msg = lmc.print_msg()
  # os.remove(T_GC_txt)
  # os.remove(ref_T_WC_txt)
  return msg


def compute_loc_metrics_odom(T_GI_txt, ref_T_W_Cp_txt): # pylint: disable=invalid-name
  """

  :param T_GI_txt: map_align_reloccheck_debug.txt
  :param ref_T_W_Cp: pose_fisheye_tf
  :param T_LI_json: calibration_optimized.json for hokuyo lidar
  :return:
  """
  output_dir = os.path.dirname(T_GI_txt)
  mat_T_IC = np.array([[0.999908, -0.0130974, -0.00351362, 0.00681938],  # pylint: disable=invalid-name
                       [0.0130978, 0.999914, 7.85796e-05, 0.00237952],
                       [0.00351229, -0.000124593, 0.999994, 2.37739e-05],
                       [0, 0, 0, 1]])

  # localization result with the camera frame as the reference body frame
  # If T_IC is very close to Identity, this step may be extraneous
  timed_marker_poses = load_and_convert_poses(
      T_GI_txt, skip_rows=1,
      left_multiplier=None, right_multiplier=mat_T_IC)

  T_GC_txt = os.path.join(output_dir, "temp_T_GC.txt")  # pylint: disable=invalid-name
  vlf.save_time_trans_quat_txt(timed_marker_poses, T_GC_txt,
                               line_format=0, delimiter=" ")

  chunk = vlf.load_time_trans_quat_txt(ref_T_W_Cp_txt, 1)
  ref_T_WC = tf_helpers.right_multiply_transform( # pylint: disable=invalid-name
      chunk, vlf.CAMERA_T_PSEUDOCAMERA.transpose())
  ref_T_WC_txt = os.path.join(output_dir, "temp_ref_T_WC.txt") # pylint: disable=invalid-name
  vlf.save_time_trans_quat_txt(ref_T_WC, ref_T_WC_txt, 0)

  lmc = vlf.LocalizationMetricCalculator(T_GC_txt, ref_T_WC_txt, 5.0)
  lmc.calculate()
  msg = lmc.print_msg()

  # os.remove(T_GC_txt)
  # os.remove(ref_T_WC_txt)
  return msg, lmc.std_prf, lmc.loc_time_to_dist, lmc.dist_interval


def main():
  """main function"""
  # map_folder dir of mapping dataset
  # pose_result_dir result dir of one loc dataset if recursive = 0, otherwise,
  #     the super dir of the result dirs of many loc datasets
  # calib_dir the dir to the calibration json file for the mapping dataset

  _, map_folder, pose_result_dir, calib_dir, recursive = sys.argv

  pose_dir_list = []
  output_dir = pose_result_dir
  if int(recursive):
    pose_dir_list = folder_helper.get_immediate_subdirectories(
        pose_result_dir, None)
  else:
    pose_dir_list = [pose_result_dir]

  allfigs = []
  output_pdf = os.path.join(output_dir, "all_overlay.pdf")
  title_str_list = []
  for pose_dir in pose_dir_list:
    try:
      figs, title_str = draw_traj_on_map_one_dir(
          map_folder, pose_dir, calib_dir)
      allfigs.extend(figs)
      title_str_list.append(title_str)
    except NotImplementedError as err:
      print(err)

  blockbar.saveallfigurestopdf(allfigs, output_pdf)
  output_txt = os.path.join(output_dir, "all_titles.txt")
  with open(output_txt, "w") as stream:
    for title_str in title_str_list:
      title_str = title_str.replace("\n", " ")
      stream.write("{}\n".format(title_str))

if __name__ == "__main__":
  main()
