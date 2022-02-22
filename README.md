# TurtleBot3 RL

Teaching a TurtleBot 3 to follow a track using reinforcement learning.

![Demo GIF](media/turtlebot3_rl.gif)

## Key info
- Trained a DQN on a Linear Annealled Policy based on Epsilon Greedy
- Fed the TurtleBot's camera feed to a CNN
- Negative reward for being outside the middle of the track
- Large maginitude negative reward for veering far off the track
- Large positive reward for reaching th finish line

## Replicating training
- Install ROS Noetic
- Install Gazebo
- Install OpenAI Gym, and Keras RL
- Pull this repo to your `~/catkin_ws/src` folder
- Run `catkin_make` in your `~/catkin_ws` folder
- Run `roscore` in one terminal
- Run `roslaunch turtlebot3_rl main.launch gui:=false headless:=true` in another
- Run `VISUALIZE=false rosrun turtlebot3_rl main.py` in yet a third
- 🚀
