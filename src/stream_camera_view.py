#!/usr/bin/python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()

def image_cb(data):
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    cv2.imshow('raw image', cv_image)
    cv2.waitKey(1)

def stream():
    rospy.init_node('turtlebot3_rl_camera')
    rospy.Subscriber("/camera/image_raw", Image, image_cb)
    rospy.spin()

if __name__ == "__main__":
    stream()
