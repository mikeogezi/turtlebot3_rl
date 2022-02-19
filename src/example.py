#!/usr/bin/env python3

import rospy
import rospkg
import pathlib
from gazebo import Gazebo
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
import cv2
from cv_bridge import CvBridge
import time
import tf
import numpy as np

IMG_WIDTH = 400
IMG_HEIGHT = 400


class Main():
    def __init__(self):
        rospy.init_node('turtlebot_main')
        rospy.loginfo('Starting up node...')
        self.r = rospy.Rate(20)
        self.start_time = time.time()
        rospy.on_shutdown(self.on_shutdown)

        self.gazebo = Gazebo()
        self.bridge = CvBridge()

        self.base_frame = rospy.get_param('~base_frame', '/base_link')
        self.odom_frame = rospy.get_param('~odom_frame', '/odom')
        self.cam_frame = rospy.get_param('~cam_frame','/camera/image_raw')
        self.scan_frame = rospy.get_param('~scan_frame','/scan')

        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.laser_scan = rospy.Subscriber(self.scan_frame, LaserScan, self.scan_cb)
        self.odom = rospy.Subscriber(self.odom_frame, Odometry, self.odom_cb)
        self.cam = rospy.Subscriber(self.cam_frame, Image, self.image_cb)

        self.logged_res = False
        self.latest_image = None
        self.latest_odom_reading = None
        self.latest_scan_reading = None

        mv = Twist()
        while (time.time() - self.start_time) < 30:
            mv.linear.x = 0.0
            mv.angular.z = 0.5
            self.cmd_vel.publish(mv)
            self.r.sleep()
        self.stop()
        rospy.loginfo('Stopping...')
        self.gazebo.reset_sim()
        rospy.loginfo('Resetting...')

    def scan_cb(self, msg):
        rospy.logdebug('Receiving scanning message...')
        self.latest_scan_reading = None
        self.gazebo.__un

    def odom_cb(self, msg):
        rospy.logdebug('Receiving odometry message...')
        self.latest_odom_reading = None

    def image_cb(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        if not self.logged_res:
            height, width, _ = cv_image.shape
            rospy.loginfo(
                'Original image resolution: {} x {}'.format(width, height))
            self.logged_res = True
        cv_image = cv2.resize(cv_image, (IMG_HEIGHT, IMG_WIDTH))
        self.latest_image = cv_image

    def stop(self):
        self.cmd_vel.publish(Twist())

    def on_shutdown(self):
        rospy.loginfo('Shutting down node...')


if __name__ == '__main__':
    Main()
