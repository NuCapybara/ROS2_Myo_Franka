from scipy.signal import find_peaks
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt

# Update the file path to the new location
file_path = "H_r1deg0h0.mat"

# Load the .mat file
mat_data = scipy.io.loadmat(file_path)

# Update the key based on the structure of your .mat file
data_key = "data"  # Replace with the correct key if it's different
data = mat_data[data_key]

# Convert to DataFrame and set column names
df = pd.DataFrame(
    data,
    columns=[
        "Ax",
        "Ay",
        "Az",  # Accelerometer data
        "Gx",
        "Gy",
        "Gz",  # Gyroscope data
        "Mx",
        "My",
        "Mz",  # Magnetometer data
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
        "S6",  # Sensor/Force data
        "time",  # Timestamp column
    ],
)

# Convert 'time' column to seconds if necessary
df["time"] = (df["time"] - df["time"].min()) / 1000  # Adjust if time is in milliseconds

# Ensure each column is in 1D array format for plotting
time_data = df["time"].values
ax_data = df["Ax"].values
ay_data = df["Ay"].values
az_data = df["Az"].values

# Plot Accelerometer data (Ax, Ay, Az)
# plt.figure(figsize=(12, 6))
# plt.plot(time_data, ax_data, label='Ax')
# plt.plot(time_data, ay_data, label='Ay')
# plt.plot(time_data, az_data, label='Az')
# plt.title("Accelerometer Data over Time")
# plt.xlabel("Time (s)")
# plt.ylabel("Acceleration (m/s²)")
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plot Gyroscope data (Gx, Gy, Gz)
# gx_data = df['Gx'].values
# gy_data = df['Gy'].values
# gz_data = df['Gz'].values
# plt.figure(figsize=(12, 6))
# plt.plot(time_data, gx_data, label='Gx')
# plt.plot(time_data, gy_data, label='Gy')
# plt.plot(time_data, gz_data, label='Gz')
# plt.title("Gyroscope Data over Time")
# plt.xlabel("Time (s)")
# plt.ylabel("Angular Velocity (rad/s)")
# plt.legend()
# plt.grid(True)
# plt.show()

# Plot Magnetometer data (Mx, My, Mz)
# mx_data = df['Mx'].values
# my_data = df['My'].values
# mz_data = df['Mz'].values
# plt.figure(figsize=(12, 6))
# plt.plot(time_data, mx_data, label='Mx')
# plt.plot(time_data, my_data, label='My')
# plt.plot(time_data, mz_data, label='Mz')
# plt.title("Magnetometer Data over Time")
# plt.xlabel("Time (s)")
# plt.ylabel("Magnetic Field (µT)")
# plt.legend()
# plt.grid(True)
# plt.show()

# Plot Sensor/Force Data (S1 to S6)
plt.figure(figsize=(12, 6))
for i in range(1, 7):
    sensor_data = df[f"S{i}"].values
    plt.plot(time_data, sensor_data, label=f"S{i}")
plt.title("Sensor/Force Data over Time")
plt.xlabel("Time (s)")
plt.ylabel("Force or Sensor Value")
plt.legend()
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Step 1: Find peaks in S4 to detect the end of active cycles
s4_data = df["S4"].values

# Detect peaks in S4 with a suitable height threshold to find the 8 peaks
peaks, _ = find_peaks(
    s4_data, height=500_000, distance=50
)  # Adjust height and distance as necessary

# Check if we have at least 8 peaks
if len(peaks) >= 8:
    # Use the index of the 8th peak as the cutoff point
    cutoff_idx = peaks[7]  # 8th peak index (0-based)
    print(f"8th peak detected at index: {cutoff_idx}")

    # Trim the data to include only rows up to the 8th peak
    df_trimmed = df.loc[:cutoff_idx].reset_index(drop=True)
else:
    print("Less than 8 peaks detected; using full data.")
    df_trimmed = df

# Plot the trimmed data to verify that only active cycles are included
plt.figure(figsize=(12, 6))
plt.plot(df_trimmed["time"].to_numpy(), df_trimmed["S4"].to_numpy(), label="S4")
plt.plot(df_trimmed["time"].to_numpy(), df_trimmed["S3"].to_numpy(), label="S3")
plt.plot(df_trimmed["time"].to_numpy(), df_trimmed["S1"].to_numpy(), label="S1")
plt.title("Trimmed Data for S4, S3, and S1 (up to the end of 4 cycles)")
plt.xlabel("Time (s)")
plt.ylabel("Sensor Values")
plt.legend()
plt.grid(True)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load the baseline and cycle peak detection configuration
baseline_s4 = 875900  # Set based on observed baseline for S4

# Peak detection parameters
peak_threshold = 0.95e6  # Slightly lower threshold to capture all peaks in S4
near_baseline_offset = 0.05 * baseline_s4  # Allow a small offset from the baseline

# Step 1: Detect peaks in S4 to identify cycles
s4_data = df['S4'].values
peaks, _ = find_peaks(s4_data, height=peak_threshold, distance=50)

# Ensure we have exactly four peaks for four cycles
if len(peaks) == 4:
    segments = []

    for peak_idx in peaks:
        # Define a broader window around each peak to capture the full rise and fall
        segment = df.loc[max(0, peak_idx-200):min(len(df)-1, peak_idx+200)]
        
        # Calculate gradients within the segment to detect changes
        s4_gradients = np.gradient(segment['S4'].values)

        # Find start of the peak by going backward until S4 is near baseline or gradient flattens
        start_of_peak = peak_idx
        for idx in range(peak_idx, max(0, peak_idx - 200), -1):
            if abs(segment.loc[idx, 'S4'] - baseline_s4) < near_baseline_offset or abs(s4_gradients[idx - peak_idx + 200]) < 0.01:
                start_of_peak = idx
                break

        # Find end of the peak by going forward until S4 is near baseline or gradient flattens
        end_of_peak = peak_idx
        for idx in range(peak_idx, min(len(df) - 1, peak_idx + 200)):
            if abs(segment.loc[idx, 'S4'] - baseline_s4) < near_baseline_offset or abs(s4_gradients[idx - peak_idx + 200]) < 0.01:
                end_of_peak = idx
                break

        # Extract the entire peak segment
        peak_segment = df.loc[start_of_peak:end_of_peak]
        segments.append(peak_segment)

    # Step 4: Plot each detected cycle segment to verify
    for i, segment in enumerate(segments):
        if not segment.empty:
            plt.figure(figsize=(10, 5))
            plt.plot(segment["time"].to_numpy(), segment["S4"].to_numpy(), label="S4", color='blue')
            plt.plot(segment["time"].to_numpy(), segment["S3"].to_numpy(), label="S3", color='orange')
            plt.plot(segment["time"].to_numpy(), segment["S1"].to_numpy(), label="S1", color='green')
            plt.title(f"Detected Cycle {i + 1} (Full Peak)")
            plt.xlabel("Time (s)")
            plt.ylabel("Sensor Values")
            plt.legend()
            plt.grid(True)
            plt.show()
else:
    print("Expected exactly four peaks, but detected:", len(peaks))
