from ultralytics import YOLO
import cv2
import numpy as np

def detect_max_confidence_object(model_path, image_path):
    # 加载 YOLOv8 模型
    model = YOLO(model_path)

    # 读取图片
    image = cv2.imread(image_path)

    # 图片预处理
    image = cv2.resize(image, (640, 640))  # 根据模型要求调整尺寸

    # 模型推理
    results = model(image, conf=0.3)  # conf 参数设置置信度阈值

    max_confidence = -1.0
    max_conf_box = None

    # 查找概率最大的检测框
    for result in results:
        for box in result.boxes:
            confidence = box.conf.item()
            if confidence > max_confidence:
                max_confidence = confidence
                max_conf_box = box

    if max_conf_box is not None:
        # 获取检测框坐标（左上角、右下角）
        x1, y1, x2, y2 = max_conf_box.xyxy[0].tolist()

        # 计算框的长宽
        width = x2 - x1
        height = y2 - y1

        # 计算框的中心位置
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # 绘制检测框和类别信息
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{model.names[max_conf_box.cls.item()]}: {max_confidence:.2f}"
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 保存检测后的图片
        output_image_path = 'detected_image.jpg'
        cv2.imwrite(output_image_path, image)
        print("Successfully saved")

        # 返回检测框的长宽和中心位置
        return (width, height, center_x, center_y)
    else:
        print("未检测到任何目标")
        return None

# 调用函数
model_path = 'yolov5nu.pt'  
image_path = 'ball.png'  
result = detect_max_confidence_object(model_path, image_path)

if result:
    width, height, center_x, center_y = result
    print("\n概率最大的检测框信息：")
    print(f"长宽：{width:.2f}, {height:.2f}")
    print(f"中心位置：{center_x:.2f}, {center_y:.2f}")