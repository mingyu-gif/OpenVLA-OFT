import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

# 박스 규격 및 시간 관련 라이브러리 추가
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class RobotArmClient(Node):
    def __init__(self):
        super().__init__('robot_arm_client')
        
        # 1. 택배 발송 센터(Action Client) 건설
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

    def send_goal(self, target_angles):
        self.get_logger().info('서버 연결 대기 중...')
        self._action_client.wait_for_server()

        # 2. 빈 택배 상자 가져오기
        goal_msg = FollowJointTrajectory.Goal()

        # 3. 송장에 관절 이름 적기 (🚨어제 확인한 ROS2 알파벳 순서와 100% 동일하게!)
        goal_msg.trajectory.joint_names = [
            'elbow_joint',
            'shoulder_lift_joint',
            'shoulder_pan_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        # 4. 내용물(각도) 포장하기
        point = JointTrajectoryPoint()
        point.positions = target_angles # AI가 예측한 6개 각도 쏙 넣기
        
        # 🚨 [안전 장치] 도착 목표 시간: 3초 동안 아주 천천히 스무스하게 움직이도록 설정!
        point.time_from_start = Duration(sec=3, nanosec=0) 

        # 상자에 내용물 담기
        goal_msg.trajectory.points.append(point)

        # 5. 로봇에게 택배 발송!
        self.get_logger().info('🚀 명령 발송 완료! 로봇 이동을 시작합니다...')
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    # (이 아래는 택배가 잘 도착했는지 영수증을 확인하는 알림 기능입니다)
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('❌ 로봇이 명령을 거부했습니다. (충돌 위험 등)')
            return
        self.get_logger().info('✅ 로봇이 명령을 수락했습니다! 이동 중...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        self.get_logger().info('🎉 목표 자세 도착 완료! 프로그램을 종료합니다.')
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    action_client = RobotArmClient()

    # 🎯 어제 VLA 모델이 추론해 낸 바로 그 6개의 관절 각도입니다!
    # (주의: 7번째 값인 그리퍼 '0.0'은 그리퍼 전용 컨트롤러로 따로 빼야 해서 제외했습니다)
    ai_target_angles = [
        2.11434275,
        -1.50974833,
         0.25978539,
         -2.18634838,
         -1.56541324,
         -0.38529593
    ]

    # 로봇에게 목표물 전송
    action_client.send_goal(ai_target_angles)
    
    # 노드 살려두기 (도착 영수증 받을 때까지 대기)
    rclpy.spin(action_client)

if __name__ == '__main__':
    main()