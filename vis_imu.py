import numpy as np
import matplotlib.pyplot as plt
from mcap.reader import make_reader
import struct

def read_mcap_file(file_path):
    timestamps = []
    raw_data_samples = []

    with open(file_path, "rb") as f:
        reader = make_reader(f)  # Create the reader object

        # Iterate through the streams in the MCAP file
        for schema, channel, message in reader.iter_messages(topics="/RU_myo/imu"):
            # Append message length and raw bytes for inspection
            data_length = len(message.data)
            raw_data_samples.append((data_length, message.data))
            
            # Store timestamp
            timestamps.append(message.log_time / 1e9)  # Converting nanoseconds to seconds

            # Limit to first few samples to avoid too much output
            if len(raw_data_samples) > 10: 
                break

    # Print sample raw data to inspect structure
    for i, (length, data) in enumerate(raw_data_samples):
        print(f"Sample {i + 1}: Length = {length} bytes, Raw Data = {data.hex()}")

    return timestamps

read_mcap_file("HumanElbowData/H_r1deg0h0/H_r1deg0h0_0.mcap")