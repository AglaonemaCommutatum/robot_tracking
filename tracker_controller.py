import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist

class TrackerController(Node):
    def __init__(self):
        super().__init__('tracker_controller')
        self.subscription = self.create_subscription(
            Point, '/yolo/target_point', self.target_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.img_width = 640  # 和相机话题一致
        self.center_x = self.img_width / 2

    def target_callback(self, msg):
        err = (msg.x - self.center_x) / self.center_x  # 归一化偏差
        twist = Twist()
        twist.linear.x = 0.2  # 前进速度（可调）
        twist.angular.z = -err  # 偏差越大，转弯越大
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = TrackerController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()