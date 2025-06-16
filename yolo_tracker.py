import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
import torch
import cv2
from cv_bridge import CvBridge

class YoloTracker(Node):
    def __init__(self):
        super().__init__('yolo_tracker')
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.pub = self.create_publisher(Point, '/yolo/target_point', 10)
        self.bridge = CvBridge()
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.get_logger().info('YOLOv5 loaded')

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(cv_image)
        if len(results.xyxy[0]) > 0:
            # 取第一个目标
            x1, y1, x2, y2, conf, cls = results.xyxy[0][0]
            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)
            p = Point()
            p.x = cx
            p.y = cy
            p.z = float(cls)
            self.pub.publish(p)

def main(args=None):
    rclpy.init(args=args)
    node = YoloTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()