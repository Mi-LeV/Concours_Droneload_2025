import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class ImageToVideoRecorder(Node):
    def __init__(self):
        super().__init__('image_to_video_recorder')
        self.bridge = CvBridge()
        
        # Video writer setup
        output_path = '/home/user/output_video.mp4'
        self.frame_rate = 20  # Adjust according to actual rate
        self.frame_size = (640, 480)  # Set according to image resolution
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, self.frame_rate, self.frame_size)

        # Subscription to the image topic
        self.subscription = self.create_subscription(
            Image,
            '/simple_drone/bottom/image_raw',
            self.image_callback,
            10
        )

    def image_callback(self, msg):
        try:
            # Convert the ROS Image message to an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Write the image to video
            self.video_writer.write(cv_image)
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    def stop_recording(self):
        # Release the video writer when done
        self.video_writer.release()
        self.get_logger().info("Video saved successfully!")

def main(args=None):
    rclpy.init(args=args)
    image_recorder = ImageToVideoRecorder()
    
    try:
        rclpy.spin(image_recorder)
    except KeyboardInterrupt:
        image_recorder.get_logger().info("Keyboard interrupt - stopping recording")
    finally:
        image_recorder.stop_recording()
        image_recorder.destroy_node()
        # Check if shutdown is necessary
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
