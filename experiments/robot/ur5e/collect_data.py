"""
collect_data.py

UR5e + RealSense D435 pick and place 데이터 수집 스크립트 (ROS2)

로봇은 별도의 제어 스크립트로 동작시키고,
이 스크립트는 실행 중인 로봇의 데이터를 구독하여 기록합니다.

필요 패키지:
  pip install h5py pynput scipy

ROS2 패키지:
  ros-humble-ur
  ros-humble-realsense2-camera
  ros-humble-cv-bridge
  ros-humble-tf2-ros

사용법:
  python experiments/robot/ur5e/collect_data.py \
    --out_dir ./collected_data \
    --task_description "pick up the red block and place it on the plate"

키보드 컨트롤:
  s : 에피소드 시작 (로봇 제어 스크립트와 동시에 누르기)
  e : 에피소드 종료 & 저장
  d : 에피소드 폐기
  q : 종료

저장 형식 (HDF5):
  episode_XXXX.hdf5
    observations/
      images/primary   (T, 256, 256, 3) uint8
      proprio          (T, 7) float32   -- [j0..j5, gripper]
    actions            (T, 7) float32   -- [dx, dy, dz, drx, dry, drz, gripper]
    attrs/
      task_description  str
      num_steps         int
"""

import argparse
import threading
from pathlib import Path
from threading import Lock

import cv2
import h5py
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
import tf2_ros
from scipy.spatial.transform import Rotation
from pynput import keyboard as pynput_keyboard


# ── 상수 ──────────────────────────────────────────────────────────────────────
IMAGE_SIZE = 256        # 저장 이미지 해상도
CONTROL_FREQ = 10       # Hz (데이터 기록 주기)
MIN_EPISODE_STEPS = 10  # 이 스텝 미만이면 폐기

# ROS2 토픽
IMAGE_TOPIC = "/camera/color/image_raw"
JOINT_TOPIC = "/joint_states"

# UR5e 조인트 이름 (ur_robot_driver 기본값)
UR5E_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Robotiq 2F-85 조인트 이름 (없으면 gripper=0 으로 처리)
GRIPPER_JOINT_NAME = "finger_joint"
GRIPPER_MAX_POS = 0.8   # rad (완전 닫힘)

# TF 프레임
BASE_FRAME = "base"
EEF_FRAME  = "tool0"


# ── DataCollector Node ────────────────────────────────────────────────────────

class DataCollector(Node):
    def __init__(self, out_dir: str, task_description: str):
        super().__init__("ur5e_data_collector")

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.task_description = task_description
        self.bridge = CvBridge()
        self.lock = Lock()

        # 상태
        self.recording = False
        self.episode_data: list = []
        self.episode_count = len(sorted(self.out_dir.glob("episode_*.hdf5")))
        self.prev_eef_pose: np.ndarray | None = None

        # 최신 관측값
        self.latest_image: np.ndarray | None = None
        self.latest_proprio: np.ndarray | None = None

        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribers
        self.create_subscription(Image, IMAGE_TOPIC, self._image_cb, 10)
        self.create_subscription(JointState, JOINT_TOPIC, self._joint_cb, 10)

        # 기록 타이머
        self.create_timer(1.0 / CONTROL_FREQ, self._record_step)

        self.get_logger().info("DataCollector 초기화 완료")
        self.get_logger().info(f"태스크: {task_description}")
        self.get_logger().info(f"저장 경로: {self.out_dir.resolve()}")

    # ── 콜백 ──────────────────────────────────────────────────────────────────

    def _image_cb(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        with self.lock:
            self.latest_image = img

    def _joint_cb(self, msg: JointState):
        joint_pos = []
        for name in UR5E_JOINT_NAMES:
            if name in msg.name:
                joint_pos.append(msg.position[msg.name.index(name)])
            else:
                return

        if GRIPPER_JOINT_NAME in msg.name:
            g = msg.position[msg.name.index(GRIPPER_JOINT_NAME)]
            gripper = float(np.clip(g / GRIPPER_MAX_POS, 0.0, 1.0))
        else:
            gripper = 0.0

        with self.lock:
            self.latest_proprio = np.array(joint_pos + [gripper], dtype=np.float32)

    # ── EEF 포즈 ──────────────────────────────────────────────────────────────

    def _get_eef_pose(self) -> np.ndarray | None:
        try:
            tf = self.tf_buffer.lookup_transform(
                BASE_FRAME, EEF_FRAME, rclpy.time.Time()
            )
            t = tf.transform.translation
            r = tf.transform.rotation
            pos = np.array([t.x, t.y, t.z])
            rot = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_euler("xyz")
            return np.concatenate([pos, rot]).astype(np.float32)
        except Exception:
            return None

    # ── 기록 ──────────────────────────────────────────────────────────────────

    def _record_step(self):
        if not self.recording:
            return

        with self.lock:
            if self.latest_image is None or self.latest_proprio is None:
                return
            image = self.latest_image.copy()
            proprio = self.latest_proprio.copy()

        eef_pose = self._get_eef_pose()
        if eef_pose is None:
            return

        if self.prev_eef_pose is None:
            delta = np.zeros(6, dtype=np.float32)
        else:
            delta = eef_pose - self.prev_eef_pose
        action = np.append(delta, proprio[-1]).astype(np.float32)
        self.prev_eef_pose = eef_pose.copy()

        self.episode_data.append({
            "image":   image,
            "proprio": proprio,
            "action":  action,
        })

    # ── 에피소드 제어 ─────────────────────────────────────────────────────────

    def start_episode(self):
        if self.recording:
            self.get_logger().warn("이미 녹화 중입니다.")
            return
        self.episode_data = []
        self.prev_eef_pose = None
        self.recording = True
        self.get_logger().info(f"[에피소드 {self.episode_count}] 녹화 시작")

    def end_episode(self, save: bool = True):
        if not self.recording:
            self.get_logger().warn("녹화 중이 아닙니다.")
            return
        self.recording = False

        n = len(self.episode_data)
        if save and n >= MIN_EPISODE_STEPS:
            self._save_episode()
        elif save:
            self.get_logger().warn(f"에피소드가 너무 짧습니다 ({n}스텝 < {MIN_EPISODE_STEPS}). 폐기.")
        else:
            self.get_logger().info("에피소드 폐기.")
        self.episode_data = []

    def _save_episode(self):
        path = self.out_dir / f"episode_{self.episode_count:04d}.hdf5"
        images   = np.stack([d["image"]   for d in self.episode_data])
        proprios = np.stack([d["proprio"] for d in self.episode_data])
        actions  = np.stack([d["action"]  for d in self.episode_data])

        with h5py.File(path, "w") as f:
            f.create_dataset("observations/images/primary", data=images, compression="lzf")
            f.create_dataset("observations/proprio",        data=proprios)
            f.create_dataset("actions",                     data=actions)
            f.attrs["task_description"] = self.task_description
            f.attrs["num_steps"] = len(self.episode_data)

        self.get_logger().info(
            f"[에피소드 {self.episode_count}] 저장 완료: {path} ({len(self.episode_data)}스텝)"
        )
        self.episode_count += 1


# ── 키보드 핸들러 ─────────────────────────────────────────────────────────────

def keyboard_handler(collector: DataCollector, stop_event: threading.Event):
    print("\n키보드 컨트롤:")
    print("  [s] 에피소드 시작")
    print("  [e] 에피소드 종료 & 저장")
    print("  [d] 에피소드 폐기")
    print("  [q] 종료\n")

    def on_press(key):
        try:
            ch = key.char
        except AttributeError:
            return

        if ch == "s":
            collector.start_episode()
        elif ch == "e":
            collector.end_episode(save=True)
        elif ch == "d":
            collector.end_episode(save=False)
        elif ch == "q":
            stop_event.set()
            return False

    with pynput_keyboard.Listener(on_press=on_press) as listener:
        listener.join()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="UR5e 데이터 수집 스크립트")
    parser.add_argument("--out_dir", type=str, default="./collected_data",
                        help="에피소드 저장 경로")
    parser.add_argument("--task_description", type=str,
                        default="pick up the block and place it on the target",
                        help="태스크 설명 (영어 지시문)")
    args = parser.parse_args()

    rclpy.init()
    collector = DataCollector(args.out_dir, args.task_description)

    stop_event = threading.Event()
    kb_thread = threading.Thread(
        target=keyboard_handler, args=(collector, stop_event), daemon=True
    )
    kb_thread.start()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(collector)

    try:
        while rclpy.ok() and not stop_event.is_set():
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if collector.recording:
            collector.end_episode(save=True)
        collector.destroy_node()
        rclpy.shutdown()
        print("종료.")


if __name__ == "__main__":
    main()
