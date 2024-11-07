import pandas as pd
import matplotlib.pyplot as plt
import re

# Load the CSV file
df = pd.read_csv('_RU_myo_emg.csv')

# Convert timestamp to seconds for easier interpretation (optional)
df['timestamp'] = (df['timestamp'] - df['timestamp'].min()) / 1e9

# Define a function to parse the _data field
def parse_emg_data(data_str):
    # Use regex to find all integers in the string
    return list(map(int, re.findall(r"-?\d+", data_str)))

# Apply parsing function to _data column
emg_data = df['_data'].apply(parse_emg_data)

# Expand parsed EMG data into separate columns
emg_df = pd.DataFrame(emg_data.tolist(), columns=[f'channel_{i}' for i in range(len(emg_data[0]))])

# Concatenate timestamp and expanded EMG data
df = pd.concat([df['timestamp'], emg_df], axis=1)

# Plot each channel over time
plt.figure(figsize=(12, 8))
for col in emg_df.columns:
    plt.plot(df['timestamp'], df[col], label=col)

# Add plot labels and legend
plt.title("EMG Data over Time")
plt.xlabel("Time (s)")
plt.ylabel("EMG Signal")
plt.legend(title="Channels", loc="upper right")
plt.grid(True)
plt.show()
