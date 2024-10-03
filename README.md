# Myo-teleop franka 
* This repo is created for connecting myo-armband to franka emika robot. I for myo data collection in ROS2. The data collected from Myo will be used for Franka teleop. 
# Package List
This repository has two ROS packages(still in progress)
This repository consists of several ROS packages 
- connect_myo
    - A library for handling serial connection between computer and myoarmband. Both armbands will publish the msgs on imu/emg through topics.
    - consisted of two nodes
        - RL_myo_node : responsible for lower myoband
        - RU_myo_node : responsible for upper myoband
    - Currently: for single armband it can searches for serial number automatically, but cannot do two armbands at the samee time. The collected rosbag data can be converted into csv.
    

- robot_control:
    - robot_control node responsible for handling incoming imu messages and mapping it into robot control. 