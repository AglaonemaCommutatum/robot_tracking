import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
import sys
from ultralytics import YOLO
from geometry_msgs.msg import Twist

class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',  # 替换为你的摄像头话题
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)  # 发布速度指令
        self.model = YOLO('yolov5nu.pt')  # 替换为您的模型路径

    def imgmsg_to_cv2(self, img_msg):
        dtype = np.dtype("uint8")  # 确保是8位
        dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
        
        # 创建 OpenCV 图像
        image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3),
                                dtype=dtype, buffer=img_msg.data)
        
        # 如果字节顺序不同，则进行字节交换
        if img_msg.is_bigendian == (sys.byteorder == 'little'):
            image_opencv = image_opencv.byteswap().newbyteorder()

        # 转换为 RGB 格式
        image_opencv = cv2.cvtColor(image_opencv, cv2.COLOR_BGR2RGB)

        return image_opencv
    
    def image_callback(self, msg):
        # 将 ROS 图像消息转换为 OpenCV 格式
        cv_image = self.imgmsg_to_cv2(msg)
        self.detect_objects(cv_image)

    def detect_objects(self, image):
        results = self.model(image)
        result = results[0]
        if result.boxes is not None:
            if len(result.boxes) == 0:
                self.get_logger().info("No objects detected, stopping robot.")
                self.stop_robot()
                return
            box = result.boxes[0]
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = int(box.cls[0])

            # 计算中心位置
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # 绘制边界框
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, f'{self.model.names[cls]} {conf:.2f}', (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # 计算机器人相对位置并发送移动指令
            self.track_ball(center_x, center_y, image.shape[1], image.shape[0])
        else:
            self.get_logger().info("No objects detected, stopping robot.")
            self.stop_robot()

        cv2.imshow("YOLO Detection", image)
        cv2.waitKey(1)

    def track_ball(self, center_x, center_y, img_width, img_height):
        # 计算相对位置
        self.get_logger().info(f"Tracking : Center X: {center_x}, Center Y: {center_y}")
        error_x = center_x - (img_width / 2)  # 计算偏差
        # error_y = center_y - (img_height / 2)

        # 生成运动指令
        move_cmd = Twist()
        move_cmd.linear.x = 0.2  # 前进速度
        move_cmd.angular.z = -float(error_x) / img_width  # 根据偏差调整转向

        # 发布速度指令
        self.publisher.publish(move_cmd)

    def stop_robot(self):
        move_cmd = Twist()  # 创建一个空的速度指令
        self.publisher.publish(move_cmd)  # 发布零速度指令

def main(args=None):
    rclpy.init(args=args)
    yolo_detector = YoloDetector()
    rclpy.spin(yolo_detector)
    yolo_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()