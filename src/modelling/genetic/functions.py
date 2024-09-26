import math
import numpy as np

def euclidean_distance(loc1, loc2):
    return math.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)

def vector_from_points(p1, p2):
    return [p2[0] - p1[0], p2[1] - p1[1]]

def dot_product(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]

def magnitude(v):
    return math.sqrt(v[0]**2 + v[1]**2)

def goal_player_angle(loc1, loc2, use_goal=True, output_in_degrees=False):
    # Vector from loc1 to loc2 (the direction of the pass)
    player_player = vector_from_points(loc1, loc2)
    
    if use_goal:
        # Vector from loc2 to the goal (goal is at [120, 40])
        player_goal = vector_from_points(loc2, [120, 40])
        
        # Dot product of the two vectors
        dot = dot_product(player_player, player_goal)
        # Magnitudes of the two vectors
        mag1 = magnitude(player_player)
        mag2 = magnitude(player_goal)
        
        # Calculate the cosine of the angle using the dot product formula
        cos_theta = dot / (mag1 * mag2)
        # Ensure cos_theta is within the valid range [-1, 1]
        cos_theta = max(-1, min(1, cos_theta))
        
        # Calculate the angle in radians
        theta = math.acos(cos_theta)
    else:
        # Calculate angle relative to the x-axis (0 is straight ahead, pi is straight backwards)
         theta = math.atan2(player_player[1], player_player[0])
    
    # If the output should be in degrees, convert the angle
    if output_in_degrees:
        return math.degrees(theta)
    else:
        return theta

def calculate_position(loc2, distance, angle_rad):
    """
    Calculate the (x, y) position of loc1 given the distance and angle from loc2.
    
    Parameters:
    loc2 (tuple): The reference point (x2, y2).
    distance (float): The distance between loc1 and loc2.
    angle (float): The angle between the line connecting loc1 and loc2 and the positive x-axis.
    
    Returns:
    tuple: The (x, y) coordinates of loc1.
    """
    # Calculate the x and y coordinates
    x = loc2[0] + distance * np.cos(angle_rad)
    y = loc2[1] + distance * np.sin(angle_rad)

    return x, y
