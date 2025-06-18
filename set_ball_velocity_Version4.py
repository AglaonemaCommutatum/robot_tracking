import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose, Twist

class BallMover(Node):
    def __init__(self):
        super().__init__('ball_mover')
        self.cli = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /gazebo/set_entity_state service...')
        self.set_ball_velocity()

    def set_ball_velocity(self):
        req = SetEntityState.Request()
        req.state = EntityState()
        req.state.name = 'orange_ball'
        req.state.pose = Pose()
        req.state.pose.position.x = 0.0
        req.state.pose.position.y = 0.0
        req.state.pose.position.z = 0.05
        req.state.pose.orientation.x = 0.0
        req.state.pose.orientation.y = 0.0
        req.state.pose.orientation.z = 0.0
        req.state.pose.orientation.w = 1.0
        req.state.twist = Twist()
        req.state.twist.linear.x = 0.3
        req.state.twist.linear.y = 0.0
        req.state.twist.linear.z = 0.0
        req.state.twist.angular.x = 0.0
        req.state.twist.angular.y = 0.0
        req.state.twist.angular.z = 0.0

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info('Set ball velocity success')
        else:
            self.get_logger().error('Failed to set ball velocity')

def main(args=None):
    rclpy.init(args=args)
    node = BallMover()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()