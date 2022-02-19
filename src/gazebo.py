import rospy
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetPhysicsProperties, SetPhysicsProperties


class Gazebo():
    def __init__(self):
        self.__unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.__pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.__reset = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.__configure_physics()

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

    def __configure_physics(self):
        self.__get_phys = rospy.ServiceProxy('/gazebo/get_physics_properties', GetPhysicsProperties)
        self.__set_phys = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)
        
        rospy.wait_for_service('/gazebo/get_physics_properties')
        props = self.__get_phys()
        
        props.max_update_rate = 0.
        rospy.wait_for_service('/gazebo/set_physics_properties')
        self.__set_phys(time_step=props.time_step, max_update_rate=props.max_update_rate, 
            gravity=props.gravity, ode_config=props.ode_config)
