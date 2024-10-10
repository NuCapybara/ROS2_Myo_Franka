import math

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to a quaternion.
    
    Roll is rotation around the x-axis
    Pitch is rotation around the y-axis
    Yaw is rotation around the z-axis
    """
    # Precompute cos and sin of half angles
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    # Compute quaternion components
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return x, y, z, w

# Example usage:
roll, pitch, yaw = 3.14159, 0.0, 0.0  # example Euler angles (no rotation)
x, y, z, w = euler_to_quaternion(roll, pitch, yaw)
print(f"Quaternion: x={x}, y={y}, z={z}, w={w}")

# Quaternion: x=0.9999999999991198, y=0.0, z=0.0, w=1.3267948966775328e-06
