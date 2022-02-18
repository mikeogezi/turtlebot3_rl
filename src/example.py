#!/usr/bin/env python3

import rospy
import rospkg
import pathlib
from gazebo import Gazebo
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
import cv2
from cv_bridge import CvBridge
import time

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

        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.laser_scan = rospy.Subscriber('/scan', LaserScan, self.scan_cb)
        self.cam = rospy.Subscriber('/camera/image_raw', Image, self.image_cb)

        self.logged_res = False
        self.latest_image = None
        self.latest_scan_reading = None

        mv = Twist()
        while (time.time() - self.start_time) < 5:
            mv.linear.x = 0.1
            mv.angular.z = 0.0
            self.cmd_vel.publish(mv)
            self.r.sleep()
        self.stop()
        rospy.loginfo('Stopping...')
        self.gazebo.reset_sim()
        rospy.loginfo('Resetting...')

    def scan_cb(self, msg):
        self.latest_scan_reading = None

    def image_cb(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        if not self.logged_res:
            height, width, _ = cv_image.shape
            rospy.loginfo(
                'Original image resolution: {}x{}'.format(width, height))
            self.logged_res = True
        cv_image = cv2.resize(cv_image, (IMG_HEIGHT, IMG_WIDTH))
        self.latest_image = cv_image

    def stop(self):
        self.cmd_vel.publish(Twist())

    def on_shutdown(self):
        rospy.loginfo('Shutting down node...')


if __name__ == '__main__':
    Main()
