import numpy as np
import matplotlib.pyplot as plt
from mcap.reader import make_reader
import struct

# Function to read MCAP data
def read_mcap_file(file_path):
    timestamps = []
    values = []

    with open(file_path, "rb") as f:
        reader = make_reader(f)  # Create the reader object

        # Iterate through the streams in the MCAP file
        for schema, channel, message in reader.iter_messages(topics="/RU_myo/emg"):
            # Interpret `message.data` as EMG data (assumed to be 16-bit signed integers)
            emg_data = struct.unpack('<8h', message.data[:16])  # Assuming each message contains 8 EMG channels
            timestamps.append(message.log_time / 1e9)  # Converting nanoseconds to seconds
            values.append(emg_data)

    return timestamps, values

# Function to plot the data
def plot_data(timestamps, values):
    values = np.array(values)  # Convert list of tuples to a NumPy array
    plt.figure(figsize=(10, 5))
    for i in range(values.shape[1]):
        plt.plot(timestamps, values[:, i], label=f"Channel {i+1}")
    plt.xlabel("Time (s)")
    plt.ylabel("EMG Value")
    plt.title("MCAP EMG Data Plot /RU_myo/emg")
    plt.legend()
    plt.grid()
    plt.show()

# Main execution
if __name__ == "__main__":
    file_path = "HumanElbowData/H_r1deg90h0/H_r1deg90h0_0.mcap"  # Path to your MCAP file
    timestamps, values = read_mcap_file(file_path)
    plot_data(timestamps, values)
