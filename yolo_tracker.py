import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
import sys
from ultralytics import YOLO

class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',  # 替换为你的摄像头话题
            self.image_callback,
            10
        )
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
        # 使用 Ultralytics YOLO 进行推理
        results = self.model(image)

        # 绘制检测结果
        result = results[0]  # 获取第一个结果
        if result.boxes is not None:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # 获取边界框坐标
                conf = box.conf[0]  # 获取置信度
                cls = int(box.cls[0])  # 获取类别并转换为整数

                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(image, f'{self.model.names[int(cls)]} {conf:.2f}', (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLO Detection", image)
        cv2.waitKey(1)  # 等待1毫秒，保持窗口更新

def main(args=None):
    rclpy.init(args=args)
    yolo_detector = YoloDetector()
    rclpy.spin(yolo_detector)
    yolo_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()