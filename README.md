# TurtleBot3 RL 🤖⚙️

Teaching a TurtleBot 3 to follow a track using reinforcement learning. Credits to [Kalvin](https://github.com/k----n), and [Professor Matt Taylor](https://drmatttaylor.net/) for the track, and the original 3d assets.

![Demo GIF](media/turtlebot3_rl.gif)

## Key info
- Trained a DQN on a Linear Annealled Policy based on Epsilon Greedy
- Fed the TurtleBot's camera feed to a CNN
- Negative reward for being outside the middle of the track
- Large magnitude negative reward for veering far off the track
- Large positive reward for reaching th finish line

## Replicating training
- Install ROS Noetic
- Install Gazebo
- Install OpenAI Gym and others with `pip install -r requirements.txt`
- Pull this repo to your `~/catkin_ws/src` folder
- Run `catkin_make` in your `~/catkin_ws` folder
- Run `roscore` in one terminal
- Run `roslaunch turtlebot3_rl main.launch gui:=false headless:=true` in another (configure the arguments accordingly to render the environment in Gazebo)
- Run `VISUALIZE=false rosrun turtlebot3_rl main.py` in yet a third (configure `VISUALIZE` env var accordingly to render the camera stream)
- 🚀

## Further work
- Running a proper headless setup on more capable machines (This was trained in an Ubuntu VM on my 2020 1.4GHz Macbook Pro)
- Providing other sensory input (odometry, or laser scan) from the robot to the RL algorithm
