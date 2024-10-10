import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Quaternion, Pose
from rclpy.time import Time
from moveit_wrapper.moveitapi import MoveItApi
from moveit_wrapper.grasp_planner import GraspPlan, GraspPlanner
from franka_msgs.action import Grasp
from rclpy.action import ActionServer, ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from pick_place_interface.action import EmptyAction, GraspProcess
import math

class pick_place(Node):
    def __init__(self):
        super().__init__("pick_place")

        # Define positions A and B on the desktop relative to the robot's base frame
        self.position_A = Pose(position=Point(x=0.4, y=0.1, z=0.0), orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))
        self.position_B = Pose(position=Point(x=0.2, y=0.2, z=0.0), orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))

        self.moveit_api = MoveItApi(
            self,
            "panda_link0",
            "panda_hand_tcp",
            "panda_manipulator",
            "/franka/joint_states",
        )
        self.grasp_planner = GraspPlanner(self.moveit_api, "panda_gripper/grasp")

        self.grasp_process = ActionClient(
            self, GraspProcess, "grasp_process", callback_group=ReentrantCallbackGroup()
        )

        self.q = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

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
        self.adjust_server = ActionServer(self,
                                        EmptyAction,
                                        "adjust",
                                        self.adjust_callback,
                                        callback_group=ReentrantCallbackGroup())
        self.size_x = 0.078
        self.size_y = 0.078
        # Set the downward-facing orientation for the end effector (facing ground)

        self.downward_orientation = Quaternion(x=0.9999999999991198, y=0.0, z=0.0, w=1.3267948966775328e-06)


    async def pick_callback(self, goal_handle):
        # Define the approach, grasp, and retreat poses for picking the object at position A
        approach_pose = self.create_pose(self.position_A.position, offset_z=0.30, orientation=self.downward_orientation)

        grasp_pose = self.create_pose(self.position_A.position, offset_z=0.05, orientation=self.downward_orientation)

        retreat_pose = self.create_pose(self.position_A.position, offset_z=0.50, orientation=self.downward_orientation)

        # Log the grasp pose
        self.get_logger().info(f"Grasping object at: {grasp_pose}")

        # Create the grasp plan
        grasp_plan = GraspPlan(
            approach_pose=approach_pose,
            grasp_pose=grasp_pose,
            grasp_command=Grasp.Goal(
                width=0.078,  # Open the gripper wide enough for the 78mm cube
                force=60.0,
                speed=0.05,
            ),
            retreat_pose=retreat_pose
        )

        # Debugging and checking grasp plan execution
        self.get_logger().info("Starting grasp execution")
        result = await self.grasp_planner.execute_grasp_plan(grasp_plan)
        if result.error_code != 0:
            self.get_logger().error(f"Grasp execution failed with error code: {result.error_code}")
        else:
            self.get_logger().info("Grasp execution succeeded")

        goal_handle.succeed()
        return EmptyAction.Result()
    
    async def place_callback(self, goal_handle):
        # Define the approach, place, and retreat poses for placing the object at position B
        approach_pose = self.create_pose(self.position_B.position, offset_z=0.30, orientation=self.downward_orientation)

        place_pose = self.create_pose(self.position_B.position, offset_z=0.1, orientation=self.downward_orientation)

        retreat_pose = self.create_pose(self.position_B.position, offset_z=0.50, orientation=self.downward_orientation)

        # Log the place pose
        self.get_logger().info(f"Placing object at: {place_pose}")

        # Create the grasp plan for placing
        grasp_plan = GraspPlan(
            approach_pose=approach_pose,
            grasp_pose=place_pose,
            grasp_command=Grasp.Goal(
                width=0.078,  # Open the gripper to release the 78mm cube
                force=60.0,
                speed=0.05,
            ),
            retreat_pose=retreat_pose
        )

        # Debugging and checking grasp plan execution
        self.get_logger().info("Starting place execution")
        result = await self.grasp_planner.execute_grasp_plan(grasp_plan)
        if result.error_code != 0:
            self.get_logger().error(f"Place execution failed with error code: {result.error_code}")
        else:
            self.get_logger().info("Place execution succeeded")

        goal_handle.succeed()
        return EmptyAction.Result()
    
    async def adjust_callback(self, goal_handle):
        self.get_logger().info("Adjusting camera (simulated adjust action)")
        goal_handle.succeed()
        return EmptyAction.Result()
    
    def create_pose(self, position, offset_z=0.0, orientation=None):
        if orientation is None:
            orientation = self.q  # Default orientation
        return Pose(
            position=Point(x=position.x, y=position.y, z=position.z + offset_z),
            orientation=orientation
        )



    def euler_to_quaternion(roll, pitch, yaw):
        """
        Convert Euler angles (roll, pitch, yaw) to a quaternion.
        
        Roll is rotation around the x-axis
        Pitch is rotation around the y-axis
        Yaw is rotation around the z-axis
        """
        # Precompute cos and sin of half angles
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        # Compute quaternion components
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return x, y, z, w



def main(args=None):
    rclpy.init(args=args)
    res = pick_place()
    rclpy.spin(res)
    rclpy.shutdown()
