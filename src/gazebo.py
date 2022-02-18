import rospy
from std_srvs.srv import Empty


class Gazebo():
    def __init__(self):
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

    def pause_sim(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        self.pause()

    def unpause_sim(self):
        rospy.wait_for_service('/gazebo/unpause_physics')
        self.unpause()

    def reset_sim(self):
        rospy.wait_for_service('/gazebo/reset_simulation')
        self.reset()
