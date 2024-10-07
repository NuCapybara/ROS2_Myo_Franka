import rclpy
from rclpy.node import Node
from tf2_ros.buffer import Buffer
from geometry_msgs.msg import (
    Point,
    Quaternion,
    Pose,
)
import tf2_geometry_msgs
from rclpy.time import Time
import numpy as np
from moveit_wrapper.moveitapi import MoveItApi
from moveit_wrapper.grasp_planner import GraspPlan, GraspPlanner
from franka_msgs.action import Grasp
from rclpy.action import ActionServer, ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from pick_place_interface.action import EmptyAction, GraspProcess
from tf2_ros.transform_listener import TransformListener
from franka_msgs.action import (
    Grasp,
)
from packing.pack import Packer, SimplePacker, Rect

class pick_place(Node):
    def __init__(self):
        super().__init__("pick_place")
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)

         # Define positions A and B on the desktop
        self.position_A = Pose(position=Point(x=0.4, y=0.1, z=0.0), orientation=Quaternion(x=0, y=0, z=0, w=1))
        self.position_B = Pose(position=Point(x=0.2, y=0.2, z=0.0), orientation=Quaternion(x=0, y=0, z=0, w=1))


        self.moveit_api = MoveItApi(
            self,
            "panda_link0",
            "panda_hand_tcp",
            "panda_manipulator",
            "/franka/joint_states",
        )
        self.grasp_planner = GraspPlanner(self.moveit_api, "panda_gripper/grasp")

        # self.grasp_action_client = ActionClient(self, Grasp, "panda_gripper/grasp")

        self.grasp_process = ActionClient(
            self, GraspProcess, "grasp_process", callback_group=ReentrantCallbackGroup()
        )

        self.approach_pose = Pose(
            position=Point(x=0.0, y=0.0, z=-0.16), orientation=Quaternion()
        )
        self.grasp_pose = Pose(
            position=Point(x=0.0, y=0.0, z=-0.06), orientation=Quaternion()
        )
        self.retreat_pose = Pose(
            position=Point(x=0.0, y=0.0, z=-0.16), orientation=Quaternion()
        )

        self.q = Quaternion(x=0, y=0, z=0, w=1)

        self.pick_server = ActionServer(self,
                                        EmptyAction,
                                        "pick",
                                        self.pick_callback,
                                        callback_group=ReentrantCallbackGroup())
        
        self.place_serve = ActionServer(self,
                                        EmptyAction,
                                        "place",
                                        self.place_callback,
                                        callback_group=ReentrantCallbackGroup())

        self.adjust_serve = ActionServer(self,
                                        EmptyAction,
                                        "adjust",
                                        self.adjust_callback,
                                        callback_group=ReentrantCallbackGroup())

        self.size_x = 0.078
        self.size_y = 0.078
        # self.size_subscriber = self.create_subscription(
        #     Point,
        #     "/dimension",
        #     self.size_callback,
        #     10,
        # )

        self.dims = []
    
    # return the dimension of the object
    # def size_callback(self, msg):
    #     self.size_x = msg.x
    #     self.size_y = msg.y
    #     # self.get_logger().info("size_x") 
    #     # self.get_logger().info(self.size_x.__str__()) 
    #     # self.get_logger().info("size_y") 
    #     # self.get_logger().info(self.size_y.__str__()) 

    async def pick_callback(self, goal_handle):
        try:
            # Lookup the transform from the desk frame to the robot's base frame
            tf = self.buffer.lookup_transform("panda_link0", "desk_frame", Time())
            self.get_logger().info("Transform from desk to robot base received")

        except Exception as e:
            self.get_logger().error(f"Transform lookup failed: {e}")
            return
        
        # Transform the approach and grasp poses for position A
        approach_pose_desk = Pose(
            position=Point(x=self.position_A.position.x, y=self.position_A.position.y, z=self.position_A.position.z + 0.10),  # 10 cm above the object
            orientation=self.position_A.orientation
        )

        grasp_pose_desk = Pose(
            position=self.position_A.position,
            orientation=self.position_A.orientation
        )

        retreat_pose_desk = Pose(
            position=Point(x=self.position_A.position.x, y=self.position_A.position.y, z=self.position_A.position.z + 0.10),  # Retreat 10 cm above the object
            orientation=self.position_A.orientation
        )

        # Perform the transformation from desk frame to robot's base frame
        approach_pose = tf2_geometry_msgs.do_transform_pose(approach_pose_desk, tf)
        grasp_pose = tf2_geometry_msgs.do_transform_pose(grasp_pose_desk, tf)
        retreat_pose = tf2_geometry_msgs.do_transform_pose(retreat_pose_desk, tf)

        # Log the transformed grasp pose
        self.get_logger().info(f"Transformed grasp_pose: {grasp_pose}")

        # Create the grasp plan
        grasp_plan = GraspPlan(
            approach_pose=approach_pose,
            grasp_pose=grasp_pose,
            grasp_command=Grasp.Goal(
                width=0.02,  # Open the gripper wide enough for the 78mm cube
                force=50.0,
                speed=0.05,
            ),
            retreat_pose=retreat_pose
        )

        await self.grasp_planner.execute_grasp_plan(grasp_plan)
        goal_handle.succeed()
        return EmptyAction.Result()
    
    async def place_callback(self, goal_handle):
        try:
            # Lookup the transform from the desk frame to the robot's base frame
            tf = self.buffer.lookup_transform("panda_link0", "desk_frame", Time())
            self.get_logger().info("Transform from desk to robot base received")

        except Exception as e:
            self.get_logger().error(f"Transform lookup failed: {e}")
            return
        
        # Transform the approach, place, and retreat poses for position B
        approach_pose_desk = self.create_pose(self.position_A.position, offset_z=0.10)
        place_pose_desk = self.create_pose(self.position_A.position)
        retreat_pose_desk = self.create_pose(self.position_A.position, offset_z=0.10)
                

        # Perform the transformation from desk frame to robot's base frame
        approach_pose = tf2_geometry_msgs.do_transform_pose(approach_pose_desk, tf)
        place_pose = tf2_geometry_msgs.do_transform_pose(place_pose_desk, tf)
        retreat_pose = tf2_geometry_msgs.do_transform_pose(retreat_pose_desk, tf)

        # Log the transformed place pose
        self.get_logger().info(f"Transformed place_pose: {place_pose}")

        # Create the grasp plan for placing
        grasp_plan = GraspPlan(
            approach_pose=approach_pose,
            grasp_pose=place_pose,
            grasp_command=Grasp.Goal(
                width=0.08,  # Open the gripper to release the 78mm cube
                force=50.0,
                speed=0.05,
            ),
            retreat_pose=retreat_pose
        )

        await self.grasp_planner.execute_grasp_plan(grasp_plan)
        goal_handle.succeed()
        return EmptyAction.Result()
    
    def create_pose(self, position, offset_z=0.0, orientation=None):
        if orientation is None:
            orientation = self.q  # Default orientation
        return Pose(
            position=Point(x=position.x, y=position.y, z=position.z + offset_z),
            orientation=orientation
        )

    # async def adjust_callback(self, goal_handle):
    #     try:
    #         tf = self.buffer.lookup_transform("panda_link0", "object", Time())
    #         self.get_logger().info(tf.__str__()) 

    #         adjust_tf = self.buffer.lookup_transform("object", "d435i_color_optical_frame", Time())

    #     except Exception as e:
    #         self.get_logger().error("no transform")
    #         return
        
    #     relate_pose = Pose(
    #                 position=Point(x=adjust_tf._transform.translation.x, y=adjust_tf._transform.translation.y, z=adjust_tf.transform.translation.z), orientation=Quaternion()
    #             )
        
    #     grasp_pose = tf2_geometry_msgs.do_transform_pose(
    #         relate_pose, tf)
    #     self.get_logger().info("pose_adas") 
    #     self.get_logger().info(grasp_pose.position.__str__()) 

    #     await self.moveit_api.plan_async(            
    #         point=grasp_pose.position,
    #         orientation=Quaternion(x=1.0,y=0.0,z=0.0,w=0.0),
    #         execute=True
    #         )
        
    #     goal_handle.succeed()
    #     return EmptyAction.Result()   

def main(args=None):
    rclpy.init(args=args)
    res = pick_place()
    rclpy.spin(res)
    rclpy.shutdown()