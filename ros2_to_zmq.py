import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from ur_msgs.msg import IOStates  # 🔹 UR I/O 상태 메시지 타입 임포트
import zmq
import time

class JointStateBridge(Node):
    def __init__(self):
        super().__init__('joint_state_bridge_node')
        
        # 최신 값을 보관할 멤버 변수 초기화
        self.current_joints = None
        self.current_gripper = 0.0  # 기본값 닫힘(LOW)

        # 1. ROS2 /joint_states 토픽 구독 (관절 각도)
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states', 
            self.joint_callback,
            10)
            
        # 2. 🔹 ROS2 /io_and_status_controller/io_states 토픽 구독 (그리퍼 핀 상태)
        self.io_sub = self.create_subscription(
            IOStates,
            '/io_and_status_controller/io_states',
            self.io_callback,
            10)
            
        # 3. ZMQ 송신 설정 (포트 5557)
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.bind("tcp://*:5557")
        
        self.get_logger().info("🚀 ROS2 관절 & 그리퍼 상태 -> ZMQ 중계기 시작 (Port: 5557)")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    def joint_callback(self, msg):
        if len(msg.position) >= 6:
            # 안전하게 앞의 6개 관절 각도(Radian)만 추출하여 저장
            self.current_joints = list(msg.position[:6])
            # 관절 데이터가 들어올 때 ZMQ로 최종 전송 시도
            self.send_combined_data()

    def io_callback(self, msg):
        """🔹 디지털 아웃풋 0번 핀(그리퍼)의 상태를 모니터링하는 콜백"""
        try:
            # UR 드라이버는 msg.digital_out_states 리스트에 핀 정보를 담아 보냅니다.
            # 우리가 제어하는 핀은 'pin 0' 이므로 인덱스 0번을 확인합니다.
            for io_msg in msg.digital_out_states:
                if io_msg.pin == 0:
                    # state 필드값을 float 형태로 저장 (1.0 = open, 0.0 = close)
                    self.current_gripper = float(io_msg.state)
                    break
        except Exception as e:
            pass

    def send_combined_data(self):
        """관절 데이터와 그리퍼 상태를 합쳐서 ZMQ로 송신"""
        if self.current_joints is not None:
            data = {
                "joints": self.current_joints,
                "gripper": self.current_gripper
            }
            self.socket.send_json(data)
            
            # 터미널 창에 현재 송신 중인 데이터 실시간 출력
            joints_str = ", ".join([f"{j:.3f}" for j in self.current_joints])
            current_time = time.strftime("%X")
            print(f"[{current_time}] 📡 ZMQ 송신중 ➔ 관절: [{joints_str}] | 그리퍼(Pin0): {self.current_gripper:.1f}", end="\r")

def main(args=None):
    rclpy.init(args=args)
    node = JointStateBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\n🛑 중계기를 종료합니다.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()