import struct
from mcap.reader import make_reader

def parse_imu_data(file_path):
    with open(file_path, "rb") as f:
        reader = make_reader(f)
        
        for i, (schema, channel, message) in enumerate(reader.iter_messages(topics="/RU_myo/imu")):
            print(f"\nMessage {i + 1}: Length = {len(message.data)} bytes")
            
            # Parse Header
            header = struct.unpack('<6I', message.data[:24])
            print(f"Header: {header}")
            
            # Parse specific byte ranges where data seems present
            try:
                # Assume orientation data in bytes 28-40 (3 floats)
                orientation = struct.unpack('<3f', message.data[28:40])
                
                # Assume angular velocity in bytes 40-52 (3 floats)
                angular_velocity = struct.unpack('<3f', message.data[40:52])
                
                # Assume linear acceleration in bytes 52-64 (3 floats)
                linear_acceleration = struct.unpack('<3f', message.data[52:64])
                
                print(f"Orientation: {orientation}")
                print(f"Angular Velocity: {angular_velocity}")
                print(f"Linear Acceleration: {linear_acceleration}")
            except struct.error as e:
                print("Error parsing IMU data:", e)

            # Stop after first few messages for brevity
            if i >= 5:
                break

# Run the parsing function
file_path = "HumanElbowData/H_r1deg0h0/H_r1deg0h0_0.mcap"  # Replace with your file path
parse_imu_data(file_path)
