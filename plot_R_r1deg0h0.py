import matplotlib.pyplot as plt
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
            joint_state_msg = deserialize_message(message.data, JointState)

            # Extract data from the deserialized message
            timestamps.append(joint_state_msg.header.stamp.sec + joint_state_msg.header.stamp.nanosec / 1e9)
            joint_names = joint_state_msg.name  # Assuming joint names remain constant
            positions.append(joint_state_msg.position)
            velocities.append(joint_state_msg.velocity)
            efforts.append(joint_state_msg.effort)

    rclpy.shutdown()
    return timestamps, joint_names, positions, velocities, efforts

def plot_joint_data(timestamps, data, ylabel, title, joint_names):
    plt.figure(figsize=(12, 6))
    for i in range(len(data[0])):  # Assuming data has entries for all joints
        plt.plot(timestamps, [entry[i] for entry in data], label=joint_names[i])
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    file_path = "RobotJointData/h0/R_r1deg0h0/R_r1deg0h0_0.mcap"  # Replace with your MCAP file path

    # Parse MCAP data
    timestamps, joint_names, positions, velocities, efforts = read_mcap_with_rclpy(file_path)

    # Print a summary of the first few entries
    for i in range(min(5, len(timestamps))):
        print(f"Timestamp: {timestamps[i]}")
        print(f"Joint Names: {joint_names}")
        print(f"Positions: {positions[i]}")
        print(f"Velocities: {velocities[i]}")
        print(f"Efforts: {efforts[i]}\n")

    # Plot the joint state data
    plot_joint_data(timestamps, positions, "Position (rad)", "Joint Positions Over Time", joint_names)
    plot_joint_data(timestamps, velocities, "Velocity (rad/s)", "Joint Velocities Over Time", joint_names)
    plot_joint_data(timestamps, efforts, "Effort (N*m)", "Joint Efforts Over Time", joint_names)
