# Myo-Teleop Franka

This repository is designed for connecting Myo armbands to the Franka Emika robot. It includes two primary functionalities:
- **Data Collection:** Collect data from Myo armbands and the Franka robot.
- **Machine Learning Pipeline:** Use a Multimodal Variational Autoencoder (mVAE) to predict robot motion based on human signals (IMU & EMG).

---

## Package List

This repository contains 7 ROS packages:

### 1. `connect_myo`
- A library for handling the serial connection between the computer and Myo armbands. Both armbands publish IMU and EMG messages through ROS topics.
- Contains two nodes:
  - **`RL_myo_node`:** Responsible for the lower Myo armband.
  - **`RU_myo_node`:** Responsible for the upper Myo armband.
- Includes a launch file to manage connections for both Myo armbands.

### 2. `ros_myo_interfaces`
- Handles ROS messages for IMU and EMG data.

### 3. `pickmeup`
- Responsible for collecting robot joint data. Users input the start position \( A \) and end position \( B \), and the robot plans and executes the trajectory.
- Depends on the `Botrista` package for trajectory planning.

### 4. `pick_place_interface`
- Handles actions and services used by the `pickmeup` package to collect robot data.

### 5. `integrate_robot_human`
- The main pipeline for:
  - Streaming Myo armband data.
  - Processing the data.
  - Feeding data into the mVAE model.
  - Predicting robot joint positions.
- Contains one node:
  - **`robot_myo_node`**
- The mVAE class is implemented in `mVAE.py`.

### 6. `moveit2_sdk_python`
- A Python MoveIt API for the Franka robot to execute trajectories. Used for providing Franka robot joint inputs.

### 7. `Botrista`
- Franka MoveIt API.

---

## Additional Folder
- **`models/b1k_e80k_eval`:** Contains the trained mVAE model. Detailed instructions can be found within the `integrate_robot_human` package.

---

## Running the Pipeline

### **In RViz Simulation**
1. Wear the two Myo armbands.
2. Run the following commands in independent terminals. Ensure the empty message (`/program_start`) is executed last to start the pipeline.

```
ros2 launch connect_myo myo_data.launch.xml
ros2 run integrate_robot_human robot_myo_node 
ros2 launch franka_moveit_config moveit.launch.py robot_ip:=dont-care use_fake_hardware:=true
ros2 topic pub /program_start std_msgs/msg/Empty "{}"
```
### **In Real Robot Operation**
1. To run the pipeline in real franka robot, run those commands in independent terminals.
```
ros2 launch connect_myo myo_data.launch.xml
ros2 run integrate_robot_human robot_myo_node 
ros2 launch franka_moveit_config rviz.launch.py robot_ip:=panda0.robot
ros2 launch franka_moveit_config moveit.launch.py robot_ip:=panda0.robot use_rviz:=false
ros2 topic pub /program_start std_msgs/msg/Empty "{}"
```



