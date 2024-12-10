import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState
from mcap.reader import make_reader

def read_mcap_with_rclpy(file_path):
    rclpy.init()
    timestamps = []
    joint_names = []
    positions = []
    velocities = []
    efforts = []

    with open(file_path, "rb") as f:
        reader = make_reader(f)

        for schema, channel, message in reader.iter_messages(topics="/joint_states"):
            # Deserialize using rclpy
            joint_state_msg = deserialize_message(message.data, JointState)

            # Extract data from the deserialized message
            timestamps.append(joint_state_msg.header.stamp.sec + joint_state_msg.header.stamp.nanosec / 1e9)
            joint_names.append(joint_state_msg.name)
            positions.append(joint_state_msg.position)
            velocities.append(joint_state_msg.velocity)
            efforts.append(joint_state_msg.effort)

            print(f"Timestamp: {timestamps[-1]}")
            print(f"Joint Names: {joint_names[-1]}")
            print(f"Positions: {positions[-1]}")
            print(f"Velocities: {velocities[-1]}")
            print(f"Efforts: {efforts[-1]}")

    rclpy.shutdown()
    return timestamps, joint_names, positions, velocities, efforts





read_mcap_with_rclpy("RobotJointData/h0/R_r1deg0h0/R_r1deg0h0_0.mcap")