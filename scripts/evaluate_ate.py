#!/usr/bin/python
# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of TUM nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Requirements: 
# sudo apt-get install python-argparse

"""
This script computes the absolute trajectory error from the ground truth
trajectory and the estimated trajectory.
"""

import sys
import numpy
import argparse
import associate
import os

def align(model,data):
    """Align two trajectories using the method of Horn (closed-form).
    
    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)
    
    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)
    
    """
    numpy.set_printoptions(precision=3,suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)
    
    W = numpy.zeros( (3,3) )
    for column in range(model.shape[1]):
        W += numpy.outer(model_zerocentered[:,column],data_zerocentered[:,column])
    U,d,Vh = numpy.linalg.linalg.svd(W.transpose())
    S = numpy.matrix(numpy.identity( 3 ))
    if(numpy.linalg.det(U) * numpy.linalg.det(Vh)<0):
        S[2,2] = -1
    rot = U*S*Vh
    trans = data.mean(1) - rot * model.mean(1)
    
    model_aligned = rot * model + trans
    alignment_error = model_aligned - data
    
    trans_error = numpy.sqrt(numpy.sum(numpy.multiply(alignment_error,alignment_error),0)).A[0]
        
    return rot,trans,trans_error

def plot_traj(ax,stamps,traj,style,color,label):
    """
    Plot a trajectory using matplotlib. 
    
    Input:
    ax -- the plot
    stamps -- time stamps (1xn)
    traj -- trajectory (3xn)
    style -- line style
    color -- line color
    label -- plot legend
    
    """
    stamps.sort()
    interval = numpy.median([s-t for s,t in zip(stamps[1:],stamps[:-1])])
    x = []
    y = []
    last = stamps[0]
    for i in range(len(stamps)):
        if stamps[i]-last < 2*interval:
            x.append(traj[i][0])
            y.append(traj[i][1])
        elif len(x)>0:
            ax.plot(x,y,style,color=color,label=label)
            label=""
            x=[]
            y=[]
        last= stamps[i]
    if len(x)>0:
        ax.plot(x,y,style,color=color,label=label)
    if label == "lidar":
        ax.plot(x[0], y[0], 'ro', label="start")
        ax.plot(x[-1], y[-1], 'ks', label="finish")

def traveled_distance(points, dimens=None):
  """
   compute the traveled distance given a sequence of way points
   points is a numpy array of shape dimen x n where n is the number of points
   dimens is the index list of interested dimensions of points (0-based)
  """
  if dimens == None:
    distance_on_dim = points[:, 1:] - points[:, :-1]
  else:
    distance_on_dim = points[dimens, 1:] - points[dimens, :-1]
  distance_squared = distance_on_dim ** 2
  distance_sum = numpy.sum(distance_squared, axis=0)
  distance = numpy.sqrt(distance_sum)
  return numpy.sum(distance)

def list_to_array(xlist, ylist, zlist=None):
    """combine a couple of list of floats into a numpy array 2xn"""
    xn = numpy.array(xlist)
    yn = numpy.array(ylist)
    if zlist == None:
      return numpy.vstack((xn, yn))
    else:
      return numpy.vstack((xn, yn, numpy.array(zlist)))

if __name__=="__main__":
    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script computes the absolute trajectory error from the ground truth trajectory and the estimated trajectory. 
    ''')
    parser.add_argument('first_file', help='ground truth trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('second_file', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('--offset', help='time offset added to the timestamps of the second file (default: 0.0)',default=0.0)
    parser.add_argument('--scale', help='scaling factor for the second trajectory (default: 1.0)',default=1.0)
    parser.add_argument('--max_difference', help='maximally allowed time difference for matching entries (default: 0.02)',default=0.02)
    parser.add_argument('--save', help='save aligned second trajectory to disk (format: stamp2 x2 y2 z2)')
    parser.add_argument('--save_associations', help='save associated first and aligned second trajectory to disk (format: stamp1 x1 y1 z1 stamp2 x2 y2 z2)')
    parser.add_argument('--plot', help='plot the first and the aligned second trajectory to an image (format: png)')
    parser.add_argument('--verbose', help='print all evaluation data (otherwise, only the RMSE absolute translational error in meters after alignment will be printed)', action='store_true')
    parser.add_argument('--offset_file', help='the file containing the negative time offset value', default="")
    args = parser.parse_args()

    first_list = associate.read_file_list(args.first_file)
    second_list = associate.read_file_list(args.second_file)
    if args.offset_file:
        fin = open(args.offset_file, "rb")
        timeoffset = -float(fin.readline())
        fin.close()
    else:
        timeoffset = float(args.offset)
    matches = associate.associate(first_list, second_list, timeoffset, float(args.max_difference))
    if len(matches)<2:
        sys.exit("Couldn't find matching timestamp pairs between groundtruth and estimated trajectory! Did you choose the correct sequence?")


    first_xyz = numpy.matrix([[float(value) for value in first_list[a][0:3]] for a,b in matches]).transpose()
    second_xyz = numpy.matrix([[float(value)*float(args.scale) for value in second_list[b][0:3]] for a,b in matches]).transpose()
    rot,trans,trans_error = align(second_xyz,first_xyz)

    a, b = matches[0]
    q_B_H0 = first_list[a][3:]
    q_W_E0 = second_list[b][3:]
    second_xyz_aligned = rot * second_xyz + trans
    
    first_stamps = first_list.keys()
    first_stamps.sort()
    first_xyz_full = numpy.matrix([[float(value) for value in first_list[b][0:3]] for b in first_stamps]).transpose()
    
    second_stamps = second_list.keys()
    second_stamps.sort()
    second_xyz_full = numpy.matrix([[float(value)*float(args.scale) for value in second_list[b][0:3]] for b in second_stamps]).transpose()
    second_xyz_full_aligned = rot * second_xyz_full + trans

    # H lidar hand, W world frame, E camera frame, H0 H frame at the start
    R_B_W = rot
    import tf
    from tf import transformations
    numpy.set_printoptions(formatter={"float_kind": lambda x: "%g" % x})
    R_H0_B = tf.transformations.quaternion_matrix(q_B_H0).transpose()
    R_H0_B = R_H0_B[:3, :3]
    R_W_E0 = tf.transformations.quaternion_matrix(q_W_E0)
    R_W_E0 = R_W_E0[:3, :3]
    R_H_E = R_H0_B * R_B_W * R_W_E0
    print("Estimated R_B_W:\n{}\nR_H_E:\n{}\nand t_B_W + avg(R_B_E)*t_E_H:{}".format(R_B_W, R_H_E, trans))

    exporttoquat = False
    if exporttoquat:
        import tf
        numpy.set_printoptions(formatter={"float_kind": lambda x: "%g" % x})
        T_H_E = tf.transformations.identity_matrix()
        T_H_E[0:3, 0:3] = R_H_E.A
        q_H_E = tf.transformations.quaternion_from_matrix(T_H_E)
        print("q_H_E(xyzw):{}".format(q_H_E))

    if args.verbose:
        print ("compared_pose_pairs %d pairs"%(len(trans_error)))
        out_path = os.path.join(os.path.dirname(args.second_file), 'alignment_result.txt')
        fout = open(out_path,'w')
        print ("absolute_translational_error.rmse %f m"%numpy.sqrt(numpy.dot(trans_error,trans_error) / len(trans_error)))
        print ("absolute_translational_error.mean %f m"%numpy.mean(trans_error))
        print ("absolute_translational_error.median %f m"%numpy.median(trans_error))
        print ("absolute_translational_error.std %f m"%numpy.std(trans_error))
        print ("absolute_translational_error.min %f m"%numpy.min(trans_error))
        print ("absolute_translational_error.max %f m"%numpy.max(trans_error))
        fout.write("absolute_translational_error.rmse(m) : " + str( numpy.  sqrt(numpy.dot(trans_error,trans_error) / len(trans_error)) ) + '\n')
        fout.write("absolute_translational_error.mean(m) : " + str (numpy.mean(trans_error)) + '\n')
        fout.write("absolute_translational_error.median(m) : " + str (numpy.median(trans_error)) + '\n')
        fout.write("absolute_translational_error.std(m) : "  + str (numpy.std(trans_error)) + '\n' )
        fout.write("absolute_translational_error.min(m) : " + str(numpy.min(trans_error)) + '\n')
        fout.write("absolute_translational_error.max(m) : " + str(numpy.max(trans_error)) + '\n')
    else:
        print ("%f"%numpy.sqrt(numpy.dot(trans_error,trans_error) / len(trans_error)))
        
    if args.save_associations:
        file = open(args.save_associations,"w")
        file.write("\n".join(["%f %f %f %f %f %f %f %f"%(a,x1,y1,z1,b,x2,y2,z2) for (a,b),(x1,y1,z1),(x2,y2,z2) in zip(matches,first_xyz.transpose().A,second_xyz_aligned.transpose().A)]))
        file.close()
        
    if args.save:
        file = open(args.save,"w")
        file.write("\n".join(["%f "%stamp+" ".join(["%f"%d for d in line]) for stamp,line in zip(second_stamps,second_xyz_full_aligned.transpose().A)]))
        file.close()

    if args.plot:
        import matplotlib
        from matplotlib.ticker import MultipleLocator, FormatStrFormatter
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        import matplotlib.pylab as pylab
        from matplotlib.patches import Ellipse
        fig = plt.figure()
        ax = fig.add_subplot(111)

        plot_traj(ax, second_stamps, second_xyz_full_aligned.transpose().A, '-', u'#ff7f0e', "vio")
        plot_traj(ax, first_stamps, first_xyz_full.transpose().A,'-',u'#1f77b4',"lidar")

        label="aligned"
        ax.legend()
        xmajorLocator   = MultipleLocator(3) 
        xmajorFormatter = FormatStrFormatter('%1.1f')  
        xminorLocator   = MultipleLocator(0.5) 

        ymajorLocator   = MultipleLocator(3)  
        ymajorFormatter = FormatStrFormatter('%1.1f')   
        yminorLocator   = MultipleLocator(0.5)
        
        first3xnarray = first_xyz_full.A
        second3xnarray = second_xyz_full_aligned.A
        
        x_min = min(numpy.min(first3xnarray[0, :]), numpy.min(second3xnarray[0, :]))*1.2
        x_max = max(numpy.max(first3xnarray[0, :]), numpy.max(second3xnarray[0, :]))*1.2
        y_min = min(numpy.min(first3xnarray[1, :]), numpy.min(second3xnarray[1, :]))*1.2
        y_max = max(numpy.max(first3xnarray[1, :]), numpy.max(second3xnarray[1, :]))*1.2

        lidardist = traveled_distance(first3xnarray, [0, 1])
        vinsdist = traveled_distance(second3xnarray, [0, 1])

        ax.axis([x_min,x_max,y_min,y_max])
        ax.set_aspect('equal', adjustable='box')
        ax.xaxis.set_major_locator(xmajorLocator)
        ax.yaxis.set_major_locator(ymajorLocator)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('Aligned lidar and vio trajectories of distances\n'
                     'lidar:{:.3f} vins:{:.3f}\n'
                     'Note gaps in trajectory mean no matches in time'.
                     format(lidardist, vinsdist))

        plt.savefig(args.plot)
        plt.show()


