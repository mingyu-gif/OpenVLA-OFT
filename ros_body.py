import rclpy, zmq
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from ur_msgs.srv import SetIO  # 그리퍼용

class BodyNode(Node):
    def __init__(self):
        super().__init__('body_node')

        # 관절 액션 클라이언트
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        self.action_client.wait_for_server()

        # 그리퍼 서비스 클라이언트
        self.gripper_client = self.create_client(SetIO, '/io_and_status_controller/set_io')
        while not self.gripper_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('그리퍼 서비스 대기 중...')

        # ZMQ
        self.zmq_context = zmq.Context()
        self.cmd_sock = self.zmq_context.socket(zmq.SUB)
        self.cmd_sock.connect("tcp://localhost:5556")
        self.cmd_sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self.cmd_sock.setsockopt(zmq.CONFLATE, 1)

        self.get_logger().info('🤖 몸통 가동 완료!')

    def send_gripper(self, gripper_cmd: float):
        """
        [반올림 직결 방식 - 매핑 오류 수정 완료]
        - cmd가 0.5 미만(예: 0.29) ➔ 반올림 값: 0.0 ➔ 0번 핀: 0.0 (닫힘 🤏)
        - cmd가 0.5 이상(예: 0.81) ➔ 반올림 값: 1.0 ➔ 0번 핀: 1.0 (열림 ✋)
        """
        req = SetIO.Request()
        req.fun   = 1    # FUN_SET_DIGITAL_OUT
        req.pin   = 0    # DOUT0
        
        # 🔹 들어온 명령을 반올림하여 0.0 또는 1.0으로 만듦
        req.state = float(round(gripper_cmd))

        # 실제 로봇에 서비스 호출 (그리퍼 구동 및 0번 핀 상태 갱신)
        future = self.gripper_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        # 🔹 로그 출력 조건문도 반올림 값(1.0=열림, 0.0=닫힘)에 맞게 확실하게 수정
        action_str = "열림 ✋" if req.state == 1.0 else "닫힘 🤏"
        self.get_logger().info(f'그리퍼 {action_str} 실행 ➔ 0번 핀 상태를 반올림 값({req.state})으로 확실하게 갱신함 (원래값={gripper_cmd:.2f})')
        
    def run_loop(self):
        JOINT_NAMES = [
            'elbow_joint', 'shoulder_lift_joint', 'shoulder_pan_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        while rclpy.ok():
            try:
                data          = self.cmd_sock.recv_json()
                target_angles = data["joints"]
                gripper_cmd   = data["gripper"]
                state         = data.get("state", "UNKNOWN")

                self.get_logger().info(f'[{state}] 🎯 수신 | 그리퍼: {gripper_cmd:.2f}')

                # 1. 그리퍼 먼저 제어
                self.send_gripper(gripper_cmd)

                # 2. 관절 이동
                goal_msg = FollowJointTrajectory.Goal()
                goal_msg.trajectory.joint_names = JOINT_NAMES

                point = JointTrajectoryPoint()
                point.positions = target_angles
                point.time_from_start = Duration(sec=1, nanosec=0)
                goal_msg.trajectory.points.append(point)

                future = self.action_client.send_goal_async(goal_msg)
                rclpy.spin_until_future_complete(self, future)
                goal_handle = future.result()

                if not goal_handle.accepted:
                    self.get_logger().error('❌ 목표 거부됨!')
                    continue

                rclpy.spin_until_future_complete(self, goal_handle.get_result_async())
                self.get_logger().info('✅ 이동 완료!')

            except Exception as e:
                self.get_logger().error(f'❌ 오류: {e}')
                continue

def main(args=None):
    rclpy.init(args=args)
    node = BodyNode()
    node.run_loop()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()