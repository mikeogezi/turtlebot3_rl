from os import stat
import rospy
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetPhysicsProperties, SetPhysicsProperties, GetModelState, SetModelState
import numpy as np
import time

class Gazebo():
    def __init__(self):
        self.__unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.__pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.__reset = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.__get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.__set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.__configure_physics()
        self.stability_wait_secs = 2.

    def pause_sim(self):
        rospy.loginfo('Pausing simulation...')
        rospy.wait_for_service('/gazebo/pause_physics')
        self.__pause()

    def unpause_sim(self):
        rospy.loginfo('Unpausing simulation...')
        rospy.wait_for_service('/gazebo/unpause_physics')
        self.__unpause()

    def reset_sim(self):
        rospy.loginfo('Resetting simulation...')
        rospy.wait_for_service('/gazebo/reset_simulation')
        self.__reset()
        r = np.random.rand()
        if r > .2 and r <= .6:
            rospy.loginfo('Resetting turtlebot to blue track...')
            self.__set_turtlebot_position_to_blue_track()
        else:
            rospy.loginfo('Resetting turtlebot to after green track...')
            self.__set_turtlebot_position_to_after_green_track()
        time.sleep(self.stability_wait_secs)

    def __configure_physics(self):
        self.__get_phys = rospy.ServiceProxy('/gazebo/get_physics_properties', GetPhysicsProperties)
        self.__set_phys = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)
        
        rospy.wait_for_service('/gazebo/get_physics_properties')
        props = self.__get_phys()
        
        props.max_update_rate = 0.
        rospy.wait_for_service('/gazebo/set_physics_properties')
        self.__set_phys(time_step=props.time_step, max_update_rate=props.max_update_rate, 
            gravity=props.gravity, ode_config=props.ode_config)

    def __set_turtlebot_position_to_blue_track(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        rospy.wait_for_service('/gazebo/set_model_state')
        
        props = self.__get_state(model_name = 'turtlebot3_burger')

        state = GetModelState()
        state.model_name = 'turtlebot3_burger'
        state.pose = props.pose
        state.twist = props.twist
        state.reference_frame = ''
        state.pose.position.x = 2.4
        state.pose.position.y = -.655
        state.pose.position.z = 0
        state.pose.orientation.w = 0.
        state.pose.orientation.x = 0.
        state.pose.orientation.y = 0.
        state.pose.orientation.z = -3.065
        state.twist.linear.x = 0.
        state.twist.linear.y = 0.
        state.twist.linear.z = 0.
        state.twist.angular.x = 0.
        state.twist.angular.y = 0.
        state.twist.angular.z = 0.

        self.__set_state(state)

    # use the values from `rosservice call /gazebo/get_model_state '{model_name: turtlebot3_burger}'` as a starting point
    def __set_turtlebot_position_to_after_green_track(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        rospy.wait_for_service('/gazebo/set_model_state')
        
        props = self.__get_state(model_name = 'turtlebot3_burger')

        state = GetModelState()
        state.model_name = 'turtlebot3_burger'
        state.pose = props.pose
        state.twist = props.twist
        state.reference_frame = ''
        state.pose.position.x = -2.19
        state.pose.position.y = 2.96
        state.pose.position.z = 0
        state.pose.orientation.w = 0.47
        state.pose.orientation.x = 0.
        state.pose.orientation.y = 0.
        state.pose.orientation.z = 0.88
        state.twist.linear.x = 0.
        state.twist.linear.y = 0.
        state.twist.linear.z = 0.
        state.twist.angular.x = 0.
        state.twist.angular.y = 0.
        state.twist.angular.z = 0.

        self.__set_state(state)
