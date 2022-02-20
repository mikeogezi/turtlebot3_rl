#!/usr/bin/env python3

import rospy
import rospkg
import gym
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from stable_baselines3.common.monitor import Monitor
from turtlebot3_env import TurtleBot3Env, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizer_v2.adam import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
import pathlib

WINDOW_LENGTH = 4

class Main():
    def __init__(self, weights_path='./weights.bin'):
        rospy.init_node('turtlebot_rl_main')
        self.r = rospy.Rate(20)

        pack = rospkg.RosPack()
        self.weights_path = str(pathlib.Path(pack.get_path('turtlebot3_rl')) / weights_path)

        self.max_steps_per_episode = 10000
        self.total_steps = 100000
        self.samp_freq = 10
        self.max_episodes = 2
        self.env = gym.make('TurtleBot3Env-v1', max_steps_per_episode=self.max_steps_per_episode, hz=self.samp_freq)
        rospy.loginfo('TurtleBot3Env-v1 env initialised')

        out = pack.get_path('turtlebot3_rl') + '/training_results'
        self.env = Monitor(self.env, out, allow_early_resets=True)
        rospy.loginfo('Env wrapped in monitor')

        self.num_actions = self.env.action_space.n
        self.dqn_model = self.build_model()

        self.load_model()
        cbs = [ModelIntervalCheckpoint(self.weights_path, verbose=1, interval=1000)]
        self.dqn_model.fit(self.env, nb_steps=self.total_steps, nb_max_episode_steps=self.max_steps_per_episode,
            visualize=str(os.environ['VISUALIZE']) == 'true', log_interval=100, callbacks=cbs)
        self.env.stop()

    def fit_manually(self):
        for episode in range(1, self.num_episodes + 1):
            self.env.reset()
            cum_reward = 0.

            for step in range(1, self.num_steps + 1):
                rospy.loginfo('Step {}'.format(step))

                action = self.env.action_space.sample()
                obs, reward, done, info = self.env.step(action)
                cum_reward += reward
                self.env.render()
                self.r.sleep()

                rospy.loginfo('Reward: {}'.format(reward))
                
                if done:
                    rospy.loginfo('Episode {} completed early'.format(episode))
                    break

            self.save_model()
            rospy.loginfo('Reward gotten during episode {}: {}'.format(episode, cum_reward))
            self.env.stop()

        self.env.close()
        
    def build_model(self):
        model = Sequential()
        input_shape = (WINDOW_LENGTH, IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS)
        model.add(Conv2D(32, (8, 8), input_shape=input_shape, strides=(4, 4), activation='relu'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (4, 4), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))

        print(model.summary())

        memory = SequentialMemory(limit=int(1e6), window_length=WINDOW_LENGTH)
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', 
            value_max=.35, value_min=.075, value_test=.05, nb_steps=self.total_steps)
        
        dqn = DQNAgent(model=model, policy=policy, gamma=.995, memory=memory,
            nb_actions=self.num_actions, train_interval=4, delta_clip=1.)
        dqn.compile(Adam(learning_rate=2.5e-4), metrics=['mae'])
        return dqn

    def build_conv_model(self):
        model = Sequential()
        input_shape = (WINDOW_LENGTH, IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS)
        model.add(Conv2D(32, (8, 8), input_shape=input_shape, strides=(4, 4)))
        model.add(MaxPooling2D())
        model.add(Conv2D(64, 2))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(2))
        model.add(Activation('tanh'))

        print(model.summary())

        memory = SequentialMemory(limit=int(1e5), window_length=4)
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', 
            value_max=1., value_min=.1, value_test=.05, nb_steps=self.total_steps)
        
        dqn = DQNAgent(model=model, policy=policy, gamma=.99, memory=memory,
            nb_steps_warmup=5, nb_actions=2)
        dqn.compile(Adam(learning_rate=2.5e-3), metrics=['mae'])
        return dqn

    def save_model(self):
        rospy.loginfo('Saving model...')
        self.dqn_model.save_weights(self.weights_path, overwrite=True)

    def load_model(self):
        rospy.loginfo('Loading model...')
        if os.path.exists(self.weights_path + '.index'):
            self.dqn_model.load_weights(self.weights_path)
            return True
        return False


if __name__ == '__main__':
    Main()
