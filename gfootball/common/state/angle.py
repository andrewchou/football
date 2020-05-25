import numpy as np

def relative_angle_bucket(delta_position):
    # Angle in radians, in the range [-pi, pi].
    radians = np.arctan2(delta_position[1], delta_position[0])
    degrees = radians * 180 / np.pi
    assert -180 <= degrees <= 180, degrees
    # print(delta_position)
    # print(degrees)
    # print((degrees + 180 + 22.5))
    # print((degrees + 180 + 22.5) // 45)
    return (int((degrees + 180 + 22.5) // 45) + 4) % 8
