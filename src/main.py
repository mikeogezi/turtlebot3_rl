#!/usr/bin/env python3

import rospy
import rospkg
import gym
import pathlib

# from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor


class Main():
    def __init__(self):
        rospy.init_node('turtlebot_rl_main')
        self.r = rospy.Rate(20)

        env = gym.make('TurtleBot3Env-v1')
        # check_env(env)
        rospy.loginfo('TurtleBot3Env-v1 env initialised and checked')

        pack = rospkg.RosPack()
        out = pathlib.Path(pack.get_path('turtlebot3_rl')) / 'training_results'
        env = Monitor(env, out)
        # check_env(env)
        rospy.loginfo('Env wrapped in monitor and checked')

        env.close()


if __name__ == '__main__':
    Main()
