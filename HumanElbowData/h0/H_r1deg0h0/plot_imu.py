import pandas as pd
import matplotlib.pyplot as plt
import re

# Load the CSV file
df = pd.read_csv('_RU_myo_imu.csv')

# Convert timestamp to seconds (optional for readability)
df['timestamp'] = (df['timestamp'] - df['timestamp'].min()) / 1e9

# Parsing helper functions
def parse_quaternion(quaternion_str):
    """Extract x, y, z, w values from Quaternion string."""
    match = re.findall(r"-?\d+\.\d+", quaternion_str)
    return list(map(float, match))

def parse_vector3(vector3_str):
    """Extract x, y, z values from Vector3 string."""
    match = re.findall(r"-?\d+\.\d+", vector3_str)
    return list(map(float, match))

# Apply parsing functions to extract orientation, angular velocity, and linear acceleration
orientation_data = df['_orientation'].apply(parse_quaternion)
angular_velocity_data = df['_angular_velocity'].apply(parse_vector3)
linear_acceleration_data = df['_linear_acceleration'].apply(parse_vector3)

# Convert parsed data into separate DataFrames with named columns
orientation_df = pd.DataFrame(orientation_data.tolist(), columns=['orientation_x', 'orientation_y', 'orientation_z', 'orientation_w'])
angular_velocity_df = pd.DataFrame(angular_velocity_data.tolist(), columns=['angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z'])
linear_acceleration_df = pd.DataFrame(linear_acceleration_data.tolist(), columns=['linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z'])

# Combine all parsed data with the timestamp
df = pd.concat([df['timestamp'], orientation_df, angular_velocity_df, linear_acceleration_df], axis=1)

# Plotting Orientation (Quaternion)
plt.figure(figsize=(12, 6))
for col in orientation_df.columns:
    plt.plot(df['timestamp'], df[col], label=col)
plt.title("Orientation (Quaternion) over Time")
plt.xlabel("Time (s)")
plt.ylabel("Orientation")
plt.legend(loc="upper right")
plt.grid(True)
plt.show()

# Plotting Angular Velocity
plt.figure(figsize=(12, 6))
for col in angular_velocity_df.columns:
    plt.plot(df['timestamp'], df[col], label=col)
plt.title("Angular Velocity over Time")
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (rad/s)")
plt.legend(loc="upper right")
plt.grid(True)
plt.show()

# Plotting Linear Acceleration
plt.figure(figsize=(12, 6))
for col in linear_acceleration_df.columns:
    plt.plot(df['timestamp'], df[col], label=col)
plt.title("Linear Acceleration over Time")
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration (m/s^2)")
plt.legend(loc="upper right")
plt.grid(True)
plt.show()
