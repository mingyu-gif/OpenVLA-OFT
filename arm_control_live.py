import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import threading

class RobotMover(Node):
    def __init__(self):
        super().__init__('robot_loop_mover')
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        self.joint_names = [
            'elbow_joint', 'shoulder_lift_joint', 'shoulder_pan_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

    def send_goal(self, angles):
        if not self._action_client.wait_for_server(timeout_sec=2.0):
            print("\n[에러] 로봇 서버 연결 실패!")
            return

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in angles]
        point.time_from_start = Duration(sec=2, nanosec=0) # 2초 동안 이동
        
        goal_msg.trajectory.points.append(point)
        print(f" >> 이동 중... {point.positions}")
        
        # 결과를 기다리지 않고 비동기로 발송
        self._action_client.send_goal_async(goal_msg)

def main():
    rclpy.init()
    mover = RobotMover()

    # ROS2 스핀을 별도 스레드에서 실행 (입력 대기 중에도 통신 유지)
    thread = threading.Thread(target=rclpy.spin, args=(mover,), daemon=True)
    thread.start()

    print("\n" + "="*50)
    print(" [UR5e 반복 제어 모드] ")
    print(" 각도 리스트를 복사해서 붙여넣으세요.")
    print(" 예시: [2.25, -1.57, -0.10, -2.22, -1.57, -0.10]")
    print(" 종료하려면 'q'를 입력하세요.")
    print("="*50)

    try:
        while True:
            user_input = input("\n목표 각도 입력: ").strip()
            
            if user_input.lower() == 'q':
                break
            
            try:
                # 괄호 [] 가 포함되어 있어도 숫자만 추출하여 리스트로 변환
                clean_input = user_input.replace('[', '').replace(']', '').replace(',', ' ')
                angles = [float(x) for x in clean_input.split()]
                
                if len(angles) != 6:
                    print(f"!! 오류: 6개의 각도가 필요합니다. (현재 {len(angles)}개)")
                    continue
                
                mover.send_goal(angles)
                
            except ValueError:
                print("!! 오류: 올바른 숫자 형식이 아닙니다.")

    except KeyboardInterrupt:
        pass
    finally:
        print("\n[*] 프로그램을 종료합니다.")
        rclpy.shutdown()

if __name__ == '__main__':
    main()