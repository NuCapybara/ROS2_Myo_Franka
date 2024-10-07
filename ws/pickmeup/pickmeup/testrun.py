import rclpy
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from rclpy.node import Node
from moveit_msgs.msg import Constraints, JointConstraint, RobotState
from trajectory_msgs.msg import JointTrajectoryPoint

class MoveGroupClient(Node):
    def __init__(self):
        super().__init__('move_group_client')
        self._action_client = ActionClient(self, MoveGroup, '/move_action')

    def send_goal(self):
        # Create a goal message
        goal_msg = MoveGroup.Goal()
        
        # Set the group name (the robot's planning group)
        goal_msg.request.group_name = 'panda_arm'

        # Define a valid goal state using joint positions
        # You can modify these positions based on your robot's configuration
        joint_constraint = JointConstraint()
        joint_constraint.joint_name = 'panda_joint1'
        joint_constraint.position = 0.0  # Example position, modify for your needs
        joint_constraint.tolerance_above = 0.1
        joint_constraint.tolerance_below = 0.1
        joint_constraint.weight = 1.0
        
        # Add additional joint constraints for other joints
        joint_constraints = [joint_constraint]

        # Create constraints and assign the joint constraints
        constraints = Constraints()
        constraints.joint_constraints = joint_constraints
        goal_msg.request.goal_constraints = [constraints]

        # Set allowed planning time
        goal_msg.request.allowed_planning_time = 5.0  # 5 seconds planning time

        # Set start state (optional, defaults to current state)
        start_state = RobotState()
        goal_msg.request.start_state = start_state

        # Send the goal
        self._action_client.wait_for_server()
        future = self._action_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        if result.error_code.val == -16:
            self.get_logger().error('Invalid motion plan: error code -16 (INVALID_MOTION_PLAN)')
        else:
            self.get_logger().info(f'Result: {result}')

def main(args=None):
    rclpy.init(args=args)

    action_client = MoveGroupClient()
    action_client.send_goal()

    rclpy.spin(action_client)

    rclpy.shutdown()

if __name__ == '__main__':
    main()
