#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021-04-16

@author: Uriel Haile Hernández Belmonte

"""

import __init__

import logging

from lifting import PoseEstimator
from lifting.utils import draw_limbs
from lifting.utils import plot_pose
#from lifting.utils import plot_pose_realtime

import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from os.path import dirname, realpath

DIR_PATH = dirname(realpath(__file__))
PROJECT_PATH = realpath(DIR_PATH + '/..')
SAVED_SESSIONS_DIR = PROJECT_PATH + '/data/saved_sessions'
SESSION_PATH = SAVED_SESSIONS_DIR + '/init_session/init'
PROB_MODEL_PATH = SAVED_SESSIONS_DIR + '/prob_model/prob_model_params.mat'

class poseEstimator:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        
        if not self.cam.isOpened():
            logging.error('No camera available')
            exit()

        # Probably is bettter to define a predefined size
        ret_val, image = self.cam.read()
        image_size = image.shape
        self.pose_estimator = PoseEstimator(image_size, SESSION_PATH, PROB_MODEL_PATH)
        self.pose_estimator.initialise()

    def plot_pose_realtime(self, pose, ax):
        """Plot the 3D pose showing the joint connections."""
        import mpl_toolkits.mplot3d.axes3d as p3

        _CONNECTION = [
            [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
            [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15],
            [15, 16]]

        def joint_color(j):
            """
            TODO: 'j' shadows name 'j' from outer scope
            """

            colors = [(0, 0, 0), (255, 0, 255), (0, 0, 255),
                    (0, 255, 255), (255, 0, 0), (0, 255, 0)]
            _c = 0
            if j in range(1, 4):
                _c = 1
            if j in range(4, 7):
                _c = 2
            if j in range(9, 11):
                _c = 3
            if j in range(11, 14):
                _c = 4
            if j in range(14, 17):
                _c = 5
            return colors[_c]

        assert (pose.ndim == 2)
        assert (pose.shape[0] == 3)
        ax.clear()

        for c in _CONNECTION:
            col = '#%02x%02x%02x' % joint_color(c[0])
            ax.plot([pose[0, c[0]], pose[0, c[1]]],
                    [pose[1, c[0]], pose[1, c[1]]],
                    [pose[2, c[0]], pose[2, c[1]]], c=col)
        for j in range(pose.shape[1]):
            col = '#%02x%02x%02x' % joint_color(j)
            ax.scatter(pose[0, j], pose[1, j], pose[2, j],
                    c=col, marker='o', edgecolor=col)
        smallest = pose.min()
        largest = pose.max()
        ax.set_xlim3d(smallest, largest)
        ax.set_ylim3d(smallest, largest)
        ax.set_zlim3d(smallest, largest)



    def __del__(self):
        if self.cam.isOpened():
            self.cam.release()





pose = poseEstimator()

fig = plt.figure()
ax = fig.gca(projection='3d')
fig.show()


while(True): 

    ret, frame = pose.cam.read() 
    
    try:
        pose_2d, visibility, pose_3d = pose.pose_estimator.estimate(frame)
        draw_limbs(frame, pose_2d, visibility)
        if(pose_3d[0].ndim == 2 or pose_3d[0].shape[0] == 3):
            logging.info('Plot pose'); 
            pose.plot_pose_realtime(pose_3d[0], ax)
            fig.canvas.draw()
            fig.canvas.flush_events()

    except ValueError:
        logging.warning('No visible people in the image. Change CENTER_TR in packages/lifting/utils/config.py ...'); 
    cv2.imshow('frame', frame) 

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
