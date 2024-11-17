import rclpy
import numpy as np
from cv_bridge import CvBridge
import time
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Vector3

from sjtu_drone_control.drone_utils.drone_object import DroneObject
from sjtu_drone_control.line_following import process_image_line, move_drone_line

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

    
    def cb_lidar(self, msg: LaserScan):
        """Callback for the lidar sensor
        Called for every Lidar message received"""
        self._lidar = msg
        #need to ensure that drone is always aligned with global frame

        wanted_heading = np.radians(0) # wanted_heading

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



    def cb_bottom_img(self, msg: Image):
        """Callback function to process the bottom camera image and control drone movement.
        Called for every bottom Image message received"""
        # Convert ROS Image message to OpenCV format
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        

        # Preprocess image to obtain yellow and green masks
        offset_x, offset_y, line_heading = process_image_line(image)

        # Move the drone if line navigation is enabled
        if LINE_NAV:
            linear_vel = move_drone_line(offset_x, offset_y, line_heading)
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