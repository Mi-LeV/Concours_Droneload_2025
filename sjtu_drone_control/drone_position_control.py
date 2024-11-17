import rclpy
import numpy as np
from cv_bridge import CvBridge
import time
import cv2
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Vector3

from sjtu_drone_control.drone_utils.drone_object import DroneObject

LIDAR_NAV = False
LINE_NAV = True

class DronePositionControl(DroneObject):
    def __init__(self):
        super().__init__('drone_position_control')

        self.bridge = CvBridge()

        self.takeOff()
        self.get_logger().info('Drone takeoff')

        self.moveTo(0.0,0.0,2.0)
        time.sleep(2)

        # Set the m_posCtrl flag to True
        #self.posCtrl(True)
        self.posCtrl(False)
        self.velMode(True)

        self.get_logger().info('Velocity control mode set to True')

        self.i = 0
        self.wanted_heading = 0
        self.timer = self.create_timer(20.0, self.my_loop)
        

    def my_loop(self):
        pos = [[0.0,0.0],[0.0,2.0],[2.0,2.0],[2.0,0.0]]
        self.wanted_heading += 90
        #self.i  = (self.i + 1) % 3
        #self.move_drone_to_pose(pos[self.i][0],pos[self.i][1],1.0)

    def move_drone_to_pose(self, x, y, z):
        # Override the move_drone_to_pose method if specific behavior is needed
        super().moveTo(x, y, z)
        self.get_logger().info(f'Moving drone to pose: x={x}, y={y}, z={z}')
    
    def cb_lidar(self, msg: LaserScan):
        """Callback for the lidar sensor
        Called for every Lidar message received"""
        self._lidar = msg
        #need to ensure that drone is always aligned with global frame

        wanted_heading = np.radians(self.wanted_heading)

        ranges = self._lidar.ranges
        obstacle_near = min(ranges) < 1
        obstacle_very_near = min(ranges) < 0.2

        if obstacle_near:
            obstacle_heading = np.arange(self._lidar.angle_min,\
             self._lidar.angle_max,self._lidar.angle_increment )[np.argmin(ranges)]

            if abs(wanted_heading - obstacle_heading) < np.radians(30):
                wanted_heading -= obstacle_heading
            speed = 0.2
        else:
            speed = 2

        if LIDAR_NAV:
            linear_vel = Vector3()
            linear_vel.x = np.cos(wanted_heading) * speed
            linear_vel.y = np.sin(wanted_heading) * speed


            self.move(v_linear=linear_vel)
            self.get_logger().info(f'Vel : {linear_vel}')

    def get_largest_contour_center(self,mask):
        """Helper to find the center of the largest contour in a binary mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest_contour)
            if moments["m00"] > 0:
                return (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
        return None
    
    def process_image_line(self,image):
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

        yellow_center = self.get_largest_contour_center(yellow_mask)
        green_center = self.get_largest_contour_center(green_mask)

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


    def move_drone_line(self,offset_x, offset_y, line_heading, speed=0.001):
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

    def cb_bottom_img(self, msg: Image):
        """Callback function to process the bottom camera image and control drone movement.
        Called for every bottom Image message received"""
        # Convert ROS Image message to OpenCV format
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        

        # Preprocess image to obtain yellow and green masks
        offset_x, offset_y, line_heading = self.process_image_line(image)

        # Move the drone if line navigation is enabled
        if LINE_NAV:
            linear_vel = self.move_drone_line(offset_x, offset_y, line_heading)
            self.move(v_linear=linear_vel)
            if line_heading is not None:
                self.get_logger().info(f'X : {offset_y} , Y : {offset_x}, H : {np.degrees(line_heading)}')


def main(args=None):
    rclpy.init(args=args)
    drone_position_control_node = DronePositionControl()
    rclpy.spin(drone_position_control_node)
    drone_position_control_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()