import rospy
import gym
import stable_baselines3 as sb3
import keras
import rl
import cv2

from gym.envs.registration import register
from gym import spaces
# from stable_baselines3.common.env_checker import check_env
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

from gazebo import Gazebo

bridge = CvBridge()

registration = register(
    id='TurtleBot3Env-v1',
    entry_point='turtlebot3_env:TurtleBot3Env',
    max_episode_steps=None,
)

IMG_HEIGHT = 400
IMG_WIDTH = 400
NUM_CHANNELS = 3


class TurtleBot3Env(gym.Env):
    def __init__(self):
        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        # self.laser_scan = rospy.Subscriber('/scan', LaserScan, self.scan_cb)
        # self.odom = rospy.Subscriber('/odom', Odometry, self.odom_cb)
        # self.cam = rospy.Service('/camera/image_raw', Image, self.image_cb)

        self.gazebo = Gazebo()
        spaces.Tuple
        self.action_space = spaces.Dict({
            'linear_x': spaces.Box(low=0., high=1., shape=(1,)),
            'angular_z': spaces.Box(low=-1., high=1., shape=(1,)),
        })
        self.environment_space = spaces.Box(
            low=0, high=255, shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))

        self.logged_res = False
        self.latest_image = None
        self.latest_odom_reading = None
        self.latest_scan_reading = None

    def scan_cb(self, msg):
        self.latest_scan_reading = None

    def odom_cb(self, msg):
        self.latest_odom_reading = None

    def image_cb(self, msg):
        cv_image = bridge.imgmsg_to_cv2(msg, 'bgr8')
        if not self.logged_res:
            height, width, _ = cv_image.shape
            rospy.loginfo(
                'Original image resolution: {}x{}'.format(width, height))
            self.logged_res = True
        cv_image = cv2.resize(cv_image, (IMG_HEIGHT, IMG_WIDTH))
        self.latest_image = cv_image

    def stop(self):
        self.cmd_vel.publish(Twist())
