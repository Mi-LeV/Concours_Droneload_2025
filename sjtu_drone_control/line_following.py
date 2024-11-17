import cv2
import numpy as np
from geometry_msgs.msg import Vector3


def get_largest_contour_center(mask):
    """Helper to find the center of the largest contour in a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest_contour)
        if moments["m00"] > 0:
            return (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
    return None

def process_image_line(image):
    """Convert the image to HSV and create masks for yellow and green colors."""
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV color ranges for yellow and green
    yellow_lower, yellow_upper = np.array([20, 100, 100]), np.array([30, 255, 255])
    green_lower, green_upper = np.array([35, 100, 100]), np.array([85, 255, 255])

    # Create masks for yellow and green
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # Apply morphological closing to reduce noise in masks
    kernel = np.ones((5, 5), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    yellow_center = get_largest_contour_center(yellow_mask)
    green_center = get_largest_contour_center(green_mask)

    image_center = (image.shape[1] // 2, image.shape[0] // 2)
    if yellow_center and green_center:
        # Find midpoint between yellow and green centers
        line_center_x = (yellow_center[0] + green_center[0]) // 2
        line_center_y = (yellow_center[1] + green_center[1]) // 2
        line_heading = np.arctan2(green_center[1] - yellow_center[1], green_center[0] - yellow_center[0])
    else:
        line_center_x, line_center_y = yellow_center or green_center or image_center
        line_heading = None

    # Calculate offset from the image center
    offset_x = line_center_x - image_center[0]

    offset_y = line_center_y - image_center[1]
    

    
    return offset_x, offset_y, line_heading


def move_drone_line(offset_x, offset_y, line_heading, speed=0.001):
    """Generate a movement command based on offsets and line heading."""
    linear_vel = Vector3()
    if line_heading is not None: # if the direction of the line is found
        far_factor = min(max(abs(offset_x), abs(offset_y)), 200) / 200 # normalise the offset between 0 and 1

        # the factor is the weight between centering the drone on the line and going down the line
        linear_vel.x = (np.cos(line_heading )) *100* speed * (1 - far_factor)\
        + (-offset_y * speed) * far_factor 
        linear_vel.y = (np.sin(line_heading))  *100* speed * (1 - far_factor)\
        + (-offset_x * speed) * far_factor
    else:
        linear_vel.x = -offset_y * speed
        linear_vel.y = -offset_x * speed
    linear_vel.z = 0.0
    return linear_vel