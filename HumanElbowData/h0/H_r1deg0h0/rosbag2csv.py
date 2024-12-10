import os
import pandas as pd
from mcap.reader import make_reader
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import sys

def rosbag2_to_csv(bag_path, output_dir):
    # Path to the MCAP file
    mcap_file_path = os.path.join(bag_path, "H_r1deg0h0_0.mcap")
    
    with open(mcap_file_path, "rb") as f:
        reader = make_reader(f)

        # Dictionary to store dataframes for each topic
        dataframes = {}

        for schema, channel, message in reader.iter_messages():
            topic_name = channel.topic
            msg_type = schema.name
            timestamp = message.log_time  # Use log_time for timestamp
            
            # Dynamically get message type and deserialize
            try:
                msg_class = get_message(msg_type)
                msg = deserialize_message(message.data, msg_class)
                msg_dict = {"timestamp": timestamp}
                
                # Convert message fields to dictionary
                msg_dict.update({field: getattr(msg, field) for field in msg.__slots__})
            except Exception as e:
                print(f"Failed to deserialize message on topic {topic_name}: {e}")
                continue

            # Accumulate data for each topic in a dataframe
            if topic_name not in dataframes:
                dataframes[topic_name] = []
            dataframes[topic_name].append(msg_dict)

        # Write each topic's data to a CSV file
        for topic_name, records in dataframes.items():
            df = pd.DataFrame(records)
            csv_filename = os.path.join(output_dir, f"{topic_name.replace('/', '_')}.csv")
            df.to_csv(csv_filename, index=False)
            print(f"Saved {topic_name} to {csv_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python rosbag2csv.py <bag_path> <output_dir>")
    else:
        bag_path = sys.argv[1]
        output_dir = sys.argv[2]
        rosbag2_to_csv(bag_path, output_dir)
