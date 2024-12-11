from moveit_wrapper.moveitapi import MoveItApi
from moveit2_sdk_python.franka_mover import FrankaMover, Moveit2Python  
from std_msgs.msg import Empty
from rclpy.callback_groups import ReentrantCallbackGroup
from pick_place_interface.action import EmptyAction
from rclpy.action import ActionClient
import numpy as np
from pick_place_interface.srv import DelayTime
from action_msgs.msg import GoalStatus
from serial.tools.list_ports import comports
from geometry_msgs.msg import Quaternion, Vector3
from sensor_msgs.msg import Imu
from std_msgs.msg import Header
from math import sqrt, degrees
import rclpy
import rclpy.callback_groups
import rclpy.exceptions
from rclpy.node import Node
from connect_myo.common import *
from connect_myo.myo_lib import *
from geometry_msgs.msg import Quaternion, Vector3
from sensor_msgs.msg import Imu
from ros_myo_interfaces.msg import MyoArm, EmgArray
from std_msgs.msg import Header
from scipy.signal import detrend
import os
sys.path.append(os.path.dirname(__file__))
from mVAE import network_param, VariationalAutoencoder, xavier_init
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from time import sleep
import asyncio
from sklearn.preprocessing import MinMaxScaler

from collections import deque

class Robot_myo(Node):
    def __init__(self):
        """Initializes the node.
        """
        super().__init__("robot_myo")

        self.ru_emg_curr = None
        self.rl_emg_curr = None
        self.ru_imu_curr = None
        self.rl_imu_curr = None
        self.batch_size = 720
        self.df_initial_size = self.batch_size + 1
        self.emg_max_size = self.df_initial_size * 4
        self.imu_max_size = self.df_initial_size
        self.ru_emg_buffer = deque(maxlen=self.emg_max_size)
        self.ru_imu_buffer = deque(maxlen=self.imu_max_size)
        self.rl_emg_buffer = deque(maxlen=self.emg_max_size)
        self.rl_imu_buffer = deque(maxlen=self.imu_max_size)
        
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.rob_scaler = MinMaxScaler(feature_range=(-1, 1))
        rob_raw_data = pd.read_csv('/home/jialuyu/Final_Project/data_collect_myo/ROS2_Myo_Franka/ws/models/b1k_e80k_eval/original_data_only_t.csv', header=None, skiprows=1).reset_index(drop=True)
        self.rob_scaler.fit(rob_raw_data.iloc[:,36:])

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.robot_joint_pred = None

        # Define model
        with self.graph.as_default():
            network_architecture = network_param()
            learning_rate = 0.00001
            self.sample_init = 0

            self.model = VariationalAutoencoder(
                self.sess,
                network_architecture,
                batch_size=self.batch_size,
                learning_rate=learning_rate,
                vae_mode=False,
                vae_mode_modalities=False
            )

        # move it api to home robot
        self.moveit_api = MoveItApi(
            self,
            "panda_link0",
            "panda_hand_tcp",
            "panda_manipulator",
            "joint_states",
            "panda",
        )
        self.franka_mover = FrankaMover()
        self.api = Moveit2Python(
            base_frame="panda_link0",
            ee_frame="panda_hand_tcp",
            group_name="panda_manipulator",
        )
        self.start_subscriber = self.create_subscription(
            Empty,
            "program_start",
            self.start_callback,
            10,
            callback_group=ReentrantCallbackGroup(),
        )


        self.ru_emg_subscriber = self.create_subscription(
            EmgArray,
            "RU_myo/emg",
            self.ru_emg_callback,
            10,
            callback_group=ReentrantCallbackGroup(),
        )

        self.rl_emg_subscriber = self.create_subscription(
            EmgArray,
            "RL_myo/emg",
            self.rl_emg_callback,
            10,
            callback_group=ReentrantCallbackGroup(),
        )

        self.ru_imu_subscriber = self.create_subscription(
            Imu,
            "RU_myo/imu",
            self.ru_imu_callback,
            10,
            callback_group=ReentrantCallbackGroup(),
        )

        self.rl_imu_subscriber = self.create_subscription(
            Imu,
            "RL_myo/imu",
            self.rl_imu_callback,
            10,
            callback_group=ReentrantCallbackGroup(),
        )


        # Delay service client
        self.delay_client = self.create_client(
            DelayTime, "delay", callback_group=ReentrantCallbackGroup()
        )

        self.timer = self.create_timer(0.5, self.timer_callback)  # Timer with 0.5s interval
        self.start = False
        
    def timer_callback(self):
        self.get_logger().info("Timer callback executed. Delay of 0.5 seconds achieved.")
        self.timer.cancel()  # Cancel the timer if it's a one-time delay

       
    def imu_process(self, imu_msg):

            """
            Extract data from an IMU message into a 1x10 NumPy array.
            
            :param imu_msg: IMU message containing orientation, angular velocity, and linear acceleration
            :return: 1x10 NumPy array
            """
            # Extract orientation (x, y, z, w)
            orientation = [
                imu_msg.orientation.x,
                imu_msg.orientation.y,
                imu_msg.orientation.z,
                imu_msg.orientation.w,
            ]
            
            # Extract angular velocity (x, y, z)
            angular_velocity = [
                imu_msg.angular_velocity.x,
                imu_msg.angular_velocity.y,
                imu_msg.angular_velocity.z,
            ]
            
            # Extract linear acceleration (x, y, z)
            linear_acceleration = [
                imu_msg.linear_acceleration.x,
                imu_msg.linear_acceleration.y,
                imu_msg.linear_acceleration.z,
            ]
            
            # Combine into a single array
            imu_array = np.array(orientation + angular_velocity + linear_acceleration)
            # self.get_logger().info(f"IMU array: {imu_array}")
            return imu_array
    
    def emg_process(self, emg_msg):
        """
        Extract data from an EMG message into a 1x8 NumPy array.
        
        :param emg_msg: EMG message containing 8 EMG values
        :return: 1x8 NumPy array
        """
        emg_array = np.array(emg_msg.data, dtype=np.int16)
        # self.get_logger().info(f"EMG array: {emg_array}")
        return emg_array

    # Store IMU and EMG data in buffers
    def ru_emg_callback(self, msg):
        self.ru_emg_curr = msg.data
        emg_processed = self.emg_process(msg)
        self.ru_emg_buffer.append(emg_processed)
        # self.get_logger().info(f"Received RU EMG data. Buffer size: {len(self.ru_emg_buffer)}")

    def rl_emg_callback(self, msg):
        self.rl_emg_curr = msg.data
        emg_processed = self.emg_process(msg)
        self.rl_emg_buffer.append(emg_processed)
        # self.get_logger().info(f"Received RL EMG data. Buffer size: {len(self.rl_emg_buffer)}")

    def ru_imu_callback(self, msg):
        self.ru_imu_curr = msg
        imu_processed = self.imu_process(msg)
        self.ru_imu_buffer.append(imu_processed)
        # self.get_logger().info(f"Received RU IMU data. Buffer size: {len(self.ru_imu_buffer)}")

    def rl_imu_callback(self, msg):
        self.rl_imu_curr = msg
        imu_processed = self.imu_process(msg)
        self.rl_imu_buffer.append(imu_processed)
        # self.get_logger().info(f"Received RL IMU data. Buffer size: {len(self.rl_imu_buffer)}")


    


    def smooth_and_rectify_multichannel(self, emg_signal, window_size=50):
        """
        Smooth and rectify a multi-channel EMG signal.

        Args:
            emg_signal (np.ndarray): Raw EMG signal (N x M array for N samples and M channels).
            window_size (int): Size of the rolling window for smoothing.

        Returns:
            np.ndarray: Processed EMG signal (N x M array).
        """
        # Ensure input is a NumPy array
        emg_signal = np.array(emg_signal)

        # Initialize an array to hold the processed EMG signal
        processed_emg = np.zeros_like(emg_signal)

        # Process each channel independently
        for channel in range(emg_signal.shape[1]):
            # Remove constant offset
            detrended_signal = detrend(emg_signal[:, channel], type='constant')
            # Full-wave rectification
            rectified_signal = np.abs(detrended_signal)
            # Smooth using a moving average
            smoothed_signal = pd.Series(rectified_signal).rolling(
                window=window_size, min_periods=1
            ).mean()
            # Store processed data
            processed_emg[:, channel] = smoothed_signal

        return processed_emg

    def build_feed_in_emgimu(self):
        # Ensure both buffers are full
        if not(len(self.rl_imu_buffer) >= self.imu_max_size and len(self.ru_imu_buffer) >= self.imu_max_size and 
                len(self.rl_emg_buffer) >= self.emg_max_size and len(self.ru_emg_buffer) >= self.emg_max_size):
            self.get_logger().info(
            f"Buffers not ready. Waiting for sufficient data. "
            f"RL IMU: {len(self.rl_imu_buffer)}, RU IMU: {len(self.ru_imu_buffer)}, "
            f"RL EMG: {len(self.rl_emg_buffer)}, RU EMG: {len(self.ru_emg_buffer)}")
            
            return None
            # self.create_timer(0.1, self.stream_to_model)

        self.get_logger().info("Buffers are ready. Proceeding with data processing.")

        # Convert buffers to NumPy arrays for processing
        rl_emg_array = np.array(self.rl_emg_buffer).reshape(self.emg_max_size, 8)  # (2884, 8)
        smoothed_rl_emg = self.smooth_and_rectify_multichannel(rl_emg_array, window_size=50)
        ru_emg_array = np.array(self.ru_emg_buffer).reshape(self.emg_max_size, 8)  # (2884, 8)
        smoothed_ru_emg = self.smooth_and_rectify_multichannel(ru_emg_array, window_size=50)

        rl_emg_array = np.array(smoothed_rl_emg).reshape(self.df_initial_size, 4, 8)
        ru_emg_array = np.array(smoothed_ru_emg).reshape(self.df_initial_size, 4, 8)   # Shape: 721 timesteps, 4 samples, 8 channels
        rl_imu_array = np.array(self.rl_imu_buffer)  # Shape: 721 timesteps, 6 channels
        ru_imu_array = np.array(self.ru_imu_buffer)  # Shape: 721 timesteps, 6 channels

        rl_emg_downsampled = rl_emg_array[:, 0, :] #downsample emg sample size to (721*8)
        ru_emg_downsampled = ru_emg_array[:, 0, :]

        if not(rl_emg_downsampled.shape[0] == self.df_initial_size and ru_emg_downsampled.shape[0] == self.df_initial_size
            and rl_imu_array.shape[0] == self.imu_max_size and ru_imu_array.shape[0] == self.imu_max_size):
            self.get_logger().info("Data not ready!! whats going on with downsample!")
            return None

        array_list = [rl_imu_array, rl_emg_downsampled, ru_imu_array, ru_emg_downsampled]
        initial_data = np.concatenate(array_list, axis=1) # (721, 36)

        prev_data = initial_data[:-1, :] # (720, 36)
        cur_data = initial_data[1:, :] # (720, 36)

        cur_prev_data_list = [cur_data, prev_data]

        def create_cur_prev_data_list(i):
            # 0: cur, 1: prev
            RL_imu = cur_prev_data_list[i][:,:10]
            RL_emg = cur_prev_data_list[i][:,10:18]
            RU_imu = cur_prev_data_list[i][:,18:28]
            RU_emg = cur_prev_data_list[i][:,28:36]

            return [RL_imu, RL_emg, RU_imu, RU_emg]

        cur_data_list = create_cur_prev_data_list(0)
        prev_data_list = create_cur_prev_data_list(1)

        item_list = []
        for i in range(len(cur_data_list)):
            item_list.append(cur_data_list[i])
            item_list.append(prev_data_list[i])
        
        combined_data = np.concatenate(item_list, axis=1) # (720, 108)
        self.scaler.fit(combined_data)
        combined_data_scaled = self.scaler.fit_transform(combined_data)
        empty_rob = np.full((self.batch_size, 36), -2)
        combined_data_feed = np.concatenate((combined_data_scaled, empty_rob), axis=1)

        with self.graph.as_default():
            x_reconstruct, _ = self.model.reconstruct(self.sess, combined_data_feed)
            x_sample_nv_1 = np.full((combined_data_feed.shape[0],10),-2)
            x_sample_nv_2 = x_reconstruct[:,:10]
            x_sample_nv_3 = np.full((combined_data_feed.shape[0],8),-2)
            x_sample_nv_4 = x_reconstruct[:,20:28]
            x_sample_nv_5 = np.full((combined_data_feed.shape[0],10),-2)
            x_sample_nv_6 = x_reconstruct[:,36:46]
            x_sample_nv_7 = np.full((combined_data_feed.shape[0],8),-2)
            x_sample_nv_8 = x_reconstruct[:,56:64]
            x_sample_nv_9 = np.full((combined_data_feed.shape[0],9),-2)
            x_sample_nv_10 = x_reconstruct[:,72:81]
            x_sample_nv_11 = np.full((combined_data_feed.shape[0],9),-2)
            x_sample_nv_12 = x_reconstruct[:,90:99]

            x_sample_list = [
                x_sample_nv_1,
                x_sample_nv_2,
                x_sample_nv_3,
                x_sample_nv_4,
                x_sample_nv_5,
                x_sample_nv_6,
                x_sample_nv_7,
                x_sample_nv_8,
                x_sample_nv_9,
                x_sample_nv_10,
                x_sample_nv_11,
                x_sample_nv_12,
            ]

            x_sample_nv = np.concatenate(x_sample_list, axis=1)
            x_pred, _ = self.model.reconstruct(self.sess, x_sample_nv)
            print(x_pred[:,72:90])

            retrieved_imu_emg = self.scaler.inverse_transform(x_pred[:,:72])
            retrieved_robot_pos = self.rob_scaler.inverse_transform(x_pred[:,72:90])
            retrieved_robot_vel = x_pred[:,90:]

            retrieved_prediction = np.concatenate(
                (
                    retrieved_imu_emg,
                    retrieved_robot_pos,
                    retrieved_robot_vel,
                ), axis=1
            )

            return retrieved_prediction

    # def stream_to_model(self):
    #     new_saver = tf.train.Saver()
    #     param_id= 1
    #     new_saver.restore(self.sess, "model/models/b1k_e80k_eval/mvae_conf_"+str(param_id)+".ckpt") ###load trained model
    #     self.get_logger.info("Model restored.")
    #     X_augm_test = self.build_feed_in_emgimu()
    #     if X_augm_test is not None: 
    #         x_reconstruct, x_reconstruct_log_sigma_sq= self.model.reconstruct(self.sess, X_augm_test)
    #         #output the robot joint from reconstructed data
    #         self.robot_joint_pred = x_reconstruct[:, 72:81]
    #         self.get_logger().info(f"Robot joint prediction: {self.robot_joint_pred}")
    #     else:
    #         self.get_logger().info("build_feed_in_emgimu returned None")
    #         return 

    # async def check_buffers(self):
    #     """Check if buffers are ready and process the data when they are."""
    #     X_augm_test = self.build_feed_in_emgimu()
    #     if X_augm_test is not None:
    #         self.get_logger().info(f"X_augm_test with IMU and EMG data: {X_augm_test}")

    #         # Perform reconstruction using the restored model
    #         x_reconstruct, x_reconstruct_log_sigma_sq = self.model.reconstruct(self.sess, X_augm_test)
    #         self.get_logger().info("Reconstruction complete.")
            
    #         # Extract robot joint predictions from reconstructed data
    #         self.robot_joint_pred = x_reconstruct[:, 72:81]
    #         self.get_logger().info(f"Robot joint prediction: {self.robot_joint_pred}")
    async def check_buffers(self):
        """Check if buffers are ready and process the data when they are."""
        X_augm_test = self.build_feed_in_emgimu()

        if X_augm_test is not None:
            self.timer.cancel()  # Stop the timer as buffers are ready
            self.get_logger().info("Buffers are ready. Proceeding with data processing.")

            # Extract robot joint predictions from reconstructed data
            self.robot_joint_pred = X_augm_test[:, 72:81]
            self.get_logger().info(f"Robot joint prediction: {self.robot_joint_pred}")
            await self.pick_place()
        else:
            self.get_logger().info(
                "Buffers not ready. Waiting for sufficient data. "
                f"RL IMU: {len(self.rl_imu_buffer)}, RU IMU: {len(self.ru_imu_buffer)}, "
                f"RL EMG: {len(self.rl_emg_buffer)}, RU EMG: {len(self.ru_emg_buffer)}"
            )

    async def stream_to_model(self):
        with self.graph.as_default():  # Ensure variables are in the graph
            new_saver = tf.train.Saver()

            # Path to the checkpoint
            param_id = 1
            checkpoint_path = f"/home/jialuyu/Final_Project/data_collect_myo/ROS2_Myo_Franka/ws/models/b1k_e80k_eval/mvae_conf_{param_id}.ckpt"

            # Restore the model session
            new_saver.restore(self.sess, checkpoint_path)
            self.get_logger().info("Model restored.")
            #NEW 
        self.timer = self.create_timer(0.1, self.check_buffers)
            


    async def start_callback(self, msg):
        """Callback function for the coffee_start subscriber. Triggers the make_coffee routine.
        """
        if not self.start:
            self.start = True

            await self.stream_to_model()
            

    # async def pick_place(self):
    #     self.get_logger().info("Starting pick and place in run node")
    #     # go to observe position
    #     print(self.robot_joint_pred)
    #     self.get_logger().info("im trying to MOVVVVVVVVVVVVVE THE ROBOT")
    #     # while not self.robot_joint_pred:
    #     #     sleep(0.1)
    #     #     self.get_logger().info("Waiting for robot joint prediction")
    #     if(self.robot_joint_pred is not None):
    #         await self.moveit_api.plan_joint_async(
    #         ["panda_joint1", "panda_joint2", "panda_joint3",
    #             "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"],
    #         self.robot_joint_pred[0], self.robot_joint_pred[1], self.robot_joint_pred[2], self.robot_joint_pred[3], self.robot_joint_pred[4], self.robot_joint_pred[5], self.robot_joint_pred[6],
    #         execute=True
    #     )
            

    #     await self.delay_client.call_async(DelayTime.Request(time=0.1))
        
       
    async def pick_place(self):
        self.get_logger().info("Starting pick and place in run node")

        if self.robot_joint_pred is not None and len(self.robot_joint_pred) > 0:
            self.get_logger().info("Robot joint prediction is available. Proceeding to move the robot.")
            print(self.robot_joint_pred.shape)
            for i, joint_values in enumerate(self.robot_joint_pred):
                if len(joint_values) >= 7:
                    joint_values_to_use = joint_values[:7]
                    self.get_logger().info(f"Moving to joint prediction {i + 1}: {joint_values_to_use}")

                    try:
                        # await self.moveit_api.plan_joint_async(
                        #     ["panda_joint1", "panda_joint2", "panda_joint3",
                        #     "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"],
                        #     joint_values_to_use,
                        #     execute=True
                        # )
                        # await self.delay_client.call_async(DelayTime.Request(time=1.0)) 
                        result = await self.franka_mover.move_by_joint(joint_values_to_use)
                        self.get_logger().info(f"Successfully executed movement for prediction {i + 1}.")
                    except Exception as e:
                        self.get_logger().error(f"Error moving to prediction {i + 1}: {e}")
                else:
                    self.get_logger().error(f"Invalid joint prediction length for row {i + 1}: {len(joint_values)}")
                    continue

            self.get_logger().info("Completed processing all joint predictions.")
        else:
            self.get_logger().info("Robot joint prediction is not available or empty.")


    # Start the machien learning model pipeline

def main(args=None):
    rclpy.init(args=args)
    res = Robot_myo()
    rclpy.spin(res)
    rclpy.shutdown()