import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger  

class ConnectionManager(Node):
    def __init__(self):
        super().__init__('connection_manager')
        self.service = self.create_service(Trigger, 'manage_connection', self.handle_connection_request)
        self.is_connected = False

    def handle_connection_request(self, request, response):
        if not self.is_connected:
            self.is_connected = True
            response.success = True
            response.message = "Connection successful."
        else:
            response.success = False
            response.message = "Connection already in use."
        return response

    def release_connection(self):
        self.is_connected = False

def main(args=None):
    rclpy.init(args=args)
    node = ConnectionManager()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
