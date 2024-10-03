#!/usr/bin/env python3
from __future__ import print_function

import argparse
import time
import math
import serial
from serial.tools.list_ports import comports

from geometry_msgs.msg import Quaternion, Vector3
from sensor_msgs.msg import Imu
from std_msgs.msg import Header
import serial
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
from example_interfaces.srv import Trigger

class RL_myo_node(Node):
    def __init__(self):
        super().__init__("rl_myo_node")
        # Client for connection conflict
        self.client = self.create_client(Trigger, 'manage_connection')
        self.try_connect()
        # Start by initializing the Myo and attempting to connect. 
        # If no Myo is found, we attempt to reconnect every 0.5 seconds

        # parser = argparse.ArgumentParser()
        # parser.add_argument('serial_port', nargs='?', default=None)

        # parser.add_argument('-i', '--imu-topic', default='myo_imu')
        # parser.add_argument('-e', '--emg-topic', default='myo_emg')
        # parser.add_argument('-a', '--arm-topic', default='myo_arm')

        # args = parser.parse_args()
        
        # Define Publishers
        self.imuPub = self.create_publisher(Imu,'RL_myo/imu', 10)
        self.emgPub = self.create_publisher(EmgArray, 'RL_myo/emg', 10)
        self.get_logger().info("I NEED M")
        
        self.m.add_emg_handler(self.proc_emg)
        self.m.add_imu_handler(self.proc_imu)

        thread_handle_read = threading.Thread(target=self.read_serial_data)
        thread_handle_read.start()
        thread_handle_read.join()


        # rospy.init_node('RL_myo_raw', anonymous=True)
    def try_connect(self):
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Connection service not available, waiting again...')
        request = Trigger.Request()
        future = self.client.call_async(request)
        future.add_done_callback(self.handle_service_response)

    def handle_service_response(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Successfully connected to Myo armband.")
                # Proceed with Myo armband connection and data handling
                        #White Myo
                # serial_port = "/dev/ttyACM1"
                arm = "RL"
                # addr = [216, 104, 114, 134, 2, 221] 
                serial_port = None
                addr = None
                
                
                print('*****')
                print(serial_port)
                print("### RL ###")
                print("Initializing...")
                print()
                
                connected = 0
                while(connected == 0):
                    try:
                        self.get_logger().info("Start trying to connect")
                        self.m = MyoRaw(serial_port, arm, addr)
                        self.get_logger().info("I got M!!!!")
                        connected = 1
                    except (ValueError, KeyboardInterrupt) as e:
                        self.get_logger().info("Myo Armband not found. Attempting to connect...")
                        self.sleep(0.5)
                        pass
            else:
                self.get_logger().info("Failed to connect: " + response.message)
        except Exception as e:
            self.get_logger().error("Service call failed %r" % (e,))

    # Package the EMG data into an EmgArray
    def proc_emg(self, emg, moving, times=[]):
        ## create an array of ints for emg data
        msg = EmgArray()
        msg.data = emg
        self.emgPub.publish(msg)
        print(emg)
        ## print framerate of received data
        times.append(time.time())
        if len(times) > 20:
            #print((len(times) - 1) / (times[-1] - times[0]))
            times.pop(0)
    # Package the IMU data into an Imu message
    def proc_imu(self, quat1, acc, gyro):
        # New info: https://github.com/thalmiclabs/myo-bluetooth/blob/master/myohw.h#L292-L295
        # Scale values for unpacking IMU data
        # define MYOHW_ORIENTATION_SCALE   16384.0f ///< See myohw_imu_data_t::orientation
        # define MYOHW_ACCELEROMETER_SCALE 2048.0f  ///< See myohw_imu_data_t::accelerometer
        # define MYOHW_GYROSCOPE_SCALE     16.0f    ///< See myohw_imu_data_t::gyroscope
        h = Header()
        h.stamp = self.get_clock().now().to_msg()
        h.frame_id = 'RL_myo'
        # We currently do not know the covariance of the sensors with each other
        cov = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        # Convert Quat to ROS2 style
        quat = Quaternion(
            x = quat1[0] / 16384.0, 
            y = quat1[1] / 16384.0, 
            z = quat1[2] / 16384.0, 
            w = quat1[3] / 16384.0,
        )
        ## Normalize the quaternion and accelerometer values
        ## TODO: check if this is shown in imu msg
        quatNorm = math.sqrt(quat.x*quat.x+quat.y*quat.y+quat.z*quat.z+quat.w*quat.w)
        normQuat = Quaternion(
            x = quat.x/quatNorm, 
            y = quat.y/quatNorm, 
            z = quat.z/quatNorm,
            w = quat.w/quatNorm,
        )
        normAcc = Vector3(
            x = acc[0]/2048.0, 
            y = acc[1]/2048.0, 
            z = acc[2]/2048.0
        )
        normGyro = Vector3(
            x = gyro[0]/16.0, 
            y = gyro[1]/16.0, 
            z = gyro[2]/16.0
        )
        # imu = Imu(h, normQuat, cov, normGyro, cov, normAcc, cov)
        imu = Imu(
            header = h,
            orientation = normQuat,
            orientation_covariance = cov, #TODO CHECK
            angular_velocity = normGyro, 
            angular_velocity_covariance = cov,#TODO CHECK
            linear_acceleration = normAcc,
            linear_acceleration_covariance = cov #TODO CHECK 
        )
        self.imuPub.publish(imu)


    def read_serial_data(self):
        self.m.connect()
        print("Connect to Myo armband")
        try:
            while True:
                packet = self.m.run()
                if packet is None:
                    self.get_logger().warn("No packet received, continuing...")
        except Exception as e:
            self.get_logger().error(f"Error in read_serial_data: {e}")
        finally:
            self.m.disconnect()


def main(args=None):
    rclpy.init(args=args)
    RL_myo_node_spin = RL_myo_node()

    try:
        rclpy.spin(RL_myo_node_spin)
    except rclpy.exceptions.ROSInterruptException:
        pass
    finally:
        RL_myo_node_spin.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()