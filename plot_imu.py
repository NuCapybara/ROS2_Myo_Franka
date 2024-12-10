import numpy as np
import matplotlib.pyplot as plt
from mcap.reader import make_reader
import struct

# Function to read the IMU data from the MCAP file
def read_imu_data(file_path):
    timestamps = []
    orientations = []
    angular_velocities = []
    linear_accelerations = []

    with open(file_path, "rb") as f:
        reader = make_reader(f)

        for i, (schema, channel, message) in enumerate(reader.iter_messages(topics="/RU_myo/imu")):
            # Extract timestamp
            timestamp = message.log_time / 1e9  # Convert from nanoseconds to seconds
            timestamps.append(timestamp)

            # Parse specific byte ranges for orientation, angular velocity, and linear acceleration
            try:
                # Assume orientation data in bytes 28-40 (3 floats)
                orientation = struct.unpack('<3f', message.data[28:40])
                orientations.append(orientation)
                
                # Assume angular velocity in bytes 40-52 (3 floats)
                angular_velocity = struct.unpack('<3f', message.data[40:52])
                angular_velocities.append(angular_velocity)
                
                # Assume linear acceleration in bytes 52-64 (3 floats)
                linear_acceleration = struct.unpack('<3f', message.data[52:64])
                linear_accelerations.append(linear_acceleration)
                
            except struct.error as e:
                print(f"Error parsing IMU data in message {i + 1}: {e}")

    return timestamps, orientations, angular_velocities, linear_accelerations

# Plotting function
def plot_imu_data(timestamps, data, labels, title, ylabel):
    plt.figure(figsize=(12, 6))
    data = np.array(data)
    for i in range(data.shape[1]):
        plt.plot(timestamps, data[:, i], label=labels[i])
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()

# Main execution
if __name__ == "__main__":
    file_path = "HumanElbowData/H_r1deg0h0/H_r1deg0h0_0.mcap"  # Replace with your file path
    timestamps, orientations, angular_velocities, linear_accelerations = read_imu_data(file_path)

    # Plot IMU data
    plot_imu_data(timestamps, orientations, ["Orientation X", "Orientation Y", "Orientation Z"], "IMU Orientation", "Orientation (quaternion/angle)")
    plot_imu_data(timestamps, angular_velocities, ["Angular Velocity X", "Angular Velocity Y", "Angular Velocity Z"], "IMU Angular Velocity", "Angular Velocity (rad/s)")
    plot_imu_data(timestamps, linear_accelerations, ["Acceleration X", "Acceleration Y", "Acceleration Z"], "IMU Linear Acceleration", "Acceleration (m/s²)")
