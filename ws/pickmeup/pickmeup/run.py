import rclpy
from rclpy.node import Node
from moveit_wrapper.moveitapi import MoveItApi
from std_msgs.msg import Empty
from rclpy.callback_groups import ReentrantCallbackGroup
from pick_place_interface.action import EmptyAction
from rclpy.action import ActionClient
import numpy as np
from pick_place_interface.srv import DelayTime


class Run(Node):
    def __init__(self):
        """Initializes the node.
        """
        super().__init__("run")

        self.action_client_pick = ActionClient(
            self, EmptyAction, "pick", callback_group=ReentrantCallbackGroup()
        )

        self.action_client_place = ActionClient(
            self, EmptyAction, "place", callback_group=ReentrantCallbackGroup()
        )

        self.action_client_adjust = ActionClient(
            self, EmptyAction, "adjust", callback_group=ReentrantCallbackGroup()
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

        self.start_subscriber = self.create_subscription(
            Empty,
            "program_start",
            self.start_callback,
            10,
            callback_group=ReentrantCallbackGroup(),
        )

        # Delay service client
        self.delay_client = self.create_client(
            DelayTime, "delay", callback_group=ReentrantCallbackGroup()
        )
        self.start = False


    async def start_callback(self, msg):
        """Callback function for the coffee_start subscriber. Triggers the make_coffee routine.
        """
        if not self.start:
            self.start = True
            await self.pick_place()

    async def pick_place(self):

        # go to observe position
        await self.moveit_api.plan_joint_async(
        ["panda_joint1", "panda_joint2", "panda_joint3",
            "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"],
        [19/180*np.pi, -50/180*np.pi, 40/180*np.pi, -122/180*np.pi, 29/180*np.pi, 81/180*np.pi, 97/180*np.pi],
        execute=True
        ) 
        await self.delay_client.call_async(DelayTime.Request(time=2.0))
        
        # adjust camera
        goal1 = EmptyAction.Goal()
        result = await self.action_client_adjust.send_goal_async(goal1)
        await result.get_result_async()

        await self.delay_client.call_async(DelayTime.Request(time=2.0))

        # pick the object
        goal2 = EmptyAction.Goal()
        result = await self.action_client_pick.send_goal_async(goal2)
        await result.get_result_async()

        # return to home position
        await self.moveit_api.go_home()

        # place the object
        goal3 = EmptyAction.Goal()
        result = await self.action_client_place.send_goal_async(goal3)
        await result.get_result_async()

        self.start = False

        # go to observe position to track the position of next object
        # await self.moveit_api.plan_joint_async(
        # ["panda_joint1", "panda_joint2", "panda_joint3",
        #     "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"],
        # [7/180*np.pi, -36/180*np.pi, 62/180*np.pi, -106/180*np.pi, 31/180*np.pi, 88/180*np.pi, 110/180*np.pi],
        # execute=True
        # ) 

def run_entry(args=None):
    rclpy.init(args=args)
    res = Run()
    rclpy.spin(res)
    rclpy.shutdown()