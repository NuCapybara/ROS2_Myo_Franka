import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import re

# Step 1: Load and Parse Cube Data
mat_data = scipy.io.loadmat('H_r1deg0h0.mat')
cube_data = mat_data.get('data')  # Ensure 'data' exists
assert cube_data is not None, "Data key 'data' not found in .mat file."

# Convert cube data to DataFrame with relevant columns
cube = pd.DataFrame(cube_data, columns=[
    'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Mx', 'My', 'Mz', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'time'
])
cube['timestamp'] = (cube['time'] - cube['time'].min()) / 1000  # Convert to seconds from start
cube = cube[['S1', 'S3', 'S4', 'timestamp']]  # Keep relevant columns

# Step 2: Load and Parse EMG Data
def parse_emg_data(data_str):
    # Use regex to extract integers
    match = re.search(r"\[(.*?)\]", data_str)
    if match:
        values = [int(x) for x in match.group(1).split(',')]
        return values
    else:
        raise ValueError(f"Cannot parse EMG data: {data_str}")

rl_emg = pd.read_csv('_RL_myo_emg.csv')
ru_emg = pd.read_csv('_RU_myo_emg.csv')
rl_emg['_data'] = rl_emg['_data'].apply(parse_emg_data)
ru_emg['_data'] = ru_emg['_data'].apply(parse_emg_data)

# Average EMG channels for peak detection
rl_emg['emg_signal'] = rl_emg['_data'].apply(np.mean)
ru_emg['emg_signal'] = ru_emg['_data'].apply(np.mean)

# Convert EMG timestamps to seconds
for df in [rl_emg, ru_emg]:
    df['timestamp'] = (df['timestamp'] - df['timestamp'].min()) / 1e9

# Step 3: Detect Peaks and Align Data
cube_peaks, _ = find_peaks(cube['S4'], height=0.5, distance=50)
cube_peak_times = cube['timestamp'].iloc[cube_peaks].values[:4]
cube = cube[cube['timestamp'] <= cube_peak_times[-1]]  # Trim extra data

# EMG peak detection
rl_emg_peaks, _ = find_peaks(rl_emg['emg_signal'], height=0.5, distance=50)
ru_emg_peaks, _ = find_peaks(ru_emg['emg_signal'], height=0.5, distance=50)

# Align start times to the first cube peak
start_time = cube_peak_times[0]
for df in [rl_emg, ru_emg, cube]:
    df['aligned_time'] = df['timestamp'] - start_time

# Step 4: Resample Data to a Common Frequency
common_frequency = 100  # Hz
end_time = min(df['aligned_time'].max() for df in [rl_emg, ru_emg, cube])
common_timestamps = np.arange(0, end_time, 1 / common_frequency)

# Interpolation function
def interpolate_to_common_time(df, new_time_index, columns, suffix):
    df_interp = pd.DataFrame({'aligned_time': new_time_index})
    for col in columns:
        df_interp[col + suffix] = np.interp(new_time_index, df['aligned_time'], df[col])
    return df_interp

# Interpolate and rename columns with suffixes
rl_emg_interp = interpolate_to_common_time(rl_emg, common_timestamps, ['emg_signal'], '_rl_emg')
ru_emg_interp = interpolate_to_common_time(ru_emg, common_timestamps, ['emg_signal'], '_ru_emg')
cube_interp = interpolate_to_common_time(cube, common_timestamps, ['S1', 'S3', 'S4'], '_cube')

# Step 5: Merge All Data on Common Timestamps
aligned_data = pd.DataFrame({'timestamp': common_timestamps})
aligned_data = aligned_data.join(rl_emg_interp.set_index('aligned_time'), on='timestamp')
aligned_data = aligned_data.join(ru_emg_interp.set_index('aligned_time'), on='timestamp')
aligned_data = aligned_data.join(cube_interp.set_index('aligned_time'), on='timestamp')

# Verify columns in aligned_data before plotting
print("Aligned data columns:", aligned_data.columns)

# Step 6: Plot Data to Check Alignment
plt.figure(figsize=(14, 8))
plt.plot(aligned_data['timestamp'], aligned_data['S4_cube'], label='S4 (Cube)')
plt.plot(aligned_data['timestamp'], aligned_data['emg_signal_rl_emg'], label='RL EMG Signal')
plt.plot(aligned_data['timestamp'], aligned_data['emg_signal_ru_emg'], label='RU EMG Signal')
plt.title("Aligned EMG and Cube Data Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Signal")
plt.legend()
plt.grid(True)
plt.show()
