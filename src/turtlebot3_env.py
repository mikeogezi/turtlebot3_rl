import rospy
import gym
import cv2
import time

from gym.envs.registration import register
from gym import spaces
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from gazebo import Gazebo
import numpy as np

registration = register(
    id='TurtleBot3Env-v1',
    entry_point='turtlebot3_env:TurtleBot3Env',
    max_episode_steps=None,
)

IMG_HEIGHT = 32
IMG_WIDTH = 32
DISP_HEIGHT = 320
DISP_WIDTH = 320
NUM_CHANNELS = 3
FINAL_REWARD = 1.

FILTERS = {
  'red': {
    'low': np.array([-10, 50, 50]),
    'high': np.array([10, 255, 255]),
  },
  'green': {
    'low': np.array([50, 50, 50]),
    'high': np.array([70, 255, 255]),
  },
  'blue': {
    'low': np.array([110, 50, 50]),
    'high': np.array([130, 255, 255]),
  },
  'white': {
    'low': np.array([0, 0, 200]),
    'high': np.array([255, 55, 255]),
  }
}



class TurtleBot3Env(gym.Env):
    def __init__(self, hz=5, max_steps_per_episode=50):
        rospy.loginfo('Starting up node...')
        self.r = rospy.Rate(hz)
        self.start_time = time.time()
        rospy.on_shutdown(self.__on_shutdown)
        self.max_steps_per_episode = max_steps_per_episode

        self.bridge = CvBridge()
        self.gazebo = Gazebo()

        self.base_frame = rospy.get_param('~base_frame', '/base_link')
        self.odom_frame = rospy.get_param('~odom_frame', '/odom')
        self.cam_frame = rospy.get_param('~cam_frame','/camera/image_raw')
        self.scan_frame = rospy.get_param('~scan_frame','/scan')

        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.laser_scan = rospy.Subscriber(self.scan_frame, LaserScan, self.__scan_cb)
        self.odom = rospy.Subscriber(self.odom_frame, Odometry, self.__odom_cb)
        self.cam = rospy.Subscriber(self.cam_frame, Image, self.__image_cb)

        self.logged_res = False
        self.latest_image = None
        self.latest_obs_image = None
        self.latest_odom_reading = None
        self.latest_scan_reading = None
        self.rendering = False
        self.action_decoder_map = None
        self.step_count = 0
        
        # self.action_space = spaces.Dict({
        #     'linear_x': spaces.Box(low=0., high=.25, shape=(1,)),
        #     'angular_z': spaces.Box(low=-.25, high=.25, shape=(1,)),
        # })
        self.linear_x_choices = 10
        self.angular_z_choices = 10
        self.action_space = spaces.Discrete(self.linear_x_choices * self.angular_z_choices)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))
        self.reward_range = (-1, 1)

    def reset(self):
        self.step_count = 0
        self.gazebo.reset_sim()
        self.gazebo.unpause_sim()
        self.__await_topic_publisher_connections()
        return self.__observe()

    def step(self, action):
        expanded_action = self.__expand_action_code(action)
        rospy.logdebug('{}: {}'.format(action, expanded_action))
        
        mv = Twist()
        mv.linear.x = expanded_action['linear_x']
        mv.angular.z = expanded_action['angular_z']
        self.cmd_vel.publish(mv)
        self.step_count += 1

        self.r.sleep()

        reward = self.__reward()
        obs = self.__observe()
        done = reward == FINAL_REWARD or self.step_count >= self.max_steps_per_episode
        info = {}

        return obs, reward, done, info

    '''
        linear_x: np.linspace(0.05, 0.5, 10)
        angular_z: np.linspace(-.25, .25, 10)
    '''
    def __expand_action_code(self, action_code):
        if self.action_decoder_map is None:
            linear_x_opts = np.linspace(0.05, 0.25, self.linear_x_choices)
            angular_z_opts = np.linspace(-.25, .25, self.angular_z_choices)
            self.action_decoder_map = np.array(np.meshgrid(linear_x_opts, angular_z_opts)).T.reshape(-1, 2)
        
        action = self.action_decoder_map[action_code]
        return {
            'linear_x': action[0],
            'angular_z': action[1],
        }

    def __reward(self):
        self.__await_image_stream()
        seen, dominance = self.__analyze_image(self.latest_image)
        major_color = self.__get_major_color(seen)
        if major_color is None:
            return -1.

        (left_dominance, right_dominance) = dominance[major_color]
        at_stop = self.__detected_stop_sign(seen)
        if at_stop:
            return FINAL_REWARD
        return -abs(left_dominance - right_dominance)

    def __get_major_color(self, seen_colors):
        max_ = 0.
        max_color = None
        keys = seen_colors.keys()
        for i in keys:
            if seen_colors[i] > max_:
                max_ = seen_colors[i]
                max_color = i
        return max_color

    def __outside_road(self, allowance=.1):
        self.__await_image_stream()
        seen, dominance = self.__analyze_image(self.latest_image)
        major_color = self.__get_major_color(seen)
        (left_dominance, right_dominance) = dominance[major_color]
        return abs(left_dominance - right_dominance) > allowance

    def __detected_stop_sign(self, seen_colors):
        return 'white' in seen_colors and 'red' in seen_colors and (seen_colors['white'] + seen_colors['red']) > .5

    def __await_image_stream(self):
        rate = rospy.Rate(5)
        while self.latest_image is None:
            rospy.logwarn_throttle_identical(5, 'Waiting to receive an image...')
            rate.sleep()

    def __observe(self):
        self.__await_image_stream()
        return self.latest_obs_image

    def render(self, mode):
        if self.latest_image is not None and mode == 'human':
            self.rendering = True
            cv2.imshow('TurtleBot3 Rendering', cv2.resize(self.latest_image, (DISP_HEIGHT, DISP_WIDTH)))
            cv2.waitKey(5)

    def __await_topic_publisher_connections(self):
        rate = rospy.Rate(5)
        while self.cmd_vel.get_num_connections() < 1:
            rospy.logwarn_throttle_identical(5, 'Waiting for a subscriber to connect to the /cmd_vel topic...')
            rate.sleep()

    def __scan_cb(self, msg):
        self.latest_scan_reading = msg

    def __odom_cb(self, msg):
        self.latest_odom_reading = msg

    def __image_cb(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        if not self.logged_res:
            height, width, _ = cv_image.shape
            rospy.loginfo(
                'Original image resolution: {} x {}'.format(width, height))
            self.logged_res = True
        self.latest_image = cv_image
        self.latest_obs_image = cv2.resize(cv_image, (IMG_HEIGHT, IMG_WIDTH))
        
    def __analyze_image(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        width_cutoff = DISP_WIDTH // 2
        height_cuttoff = DISP_HEIGHT // 2
        pixel_count = DISP_HEIGHT * DISP_WIDTH
        seen_colors = {}
        side_dominance = {}

        for color in FILTERS.keys():
            filtered_image = cv2.inRange(hsv, FILTERS[color]['low'], FILTERS[color]['high'])
            left = filtered_image[:, :width_cutoff]
            right = filtered_image[:, width_cutoff:]
            left_pct = np.sum(left == 255) / (pixel_count / 2.)
            right_pct = np.sum(right == 255) / (pixel_count / 2.)
            pct = (left_pct + right_pct) / 2.
            if (left_pct + right_pct) == 0:
                left_dominance = right_dominance = 0.5
            else:
                left_dominance = left_pct / (left_pct + right_pct)
                right_dominance = right_pct / (left_pct + right_pct)
            
            if pct > 0.01:
                rospy.logdebug('Percentage of {}: {}'.format(color, pct))
                seen_colors[color] = pct
                side_dominance[color] = (left_dominance, right_dominance)
            else:
                rospy.logdebug('Percentage of {}: {}'.format(color, pct))
        
        return (seen_colors, side_dominance)

    def stop(self):
        self.cmd_vel.publish(Twist())

    def __on_shutdown(self):
        rospy.loginfo('Shutting down node...')
