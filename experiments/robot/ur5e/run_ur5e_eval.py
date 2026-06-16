"""
run_ur5e_eval.py [최종 실로봇 연동 및 상태 기반 다중 태스크 지시문 반영 버전]

UR5e 로봇에서 OpenVLA-OFT 정책을 실행하는 inference 코드.
입력받은 그리퍼 상태에 따라 태스크 지시문(Instruction)을 동적으로 변경하며,
터미널 로그에 현재 수행 중인 태스크를 명확히 출력합니다.
"""

import json
import sys
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import zmq
from PIL import Image

# OpenVLA-OFT 경로를 최우선 순위로 추가
sys.path.insert(0, "/home/affctiv/openvla-oft")

from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_vla,
    get_vla_action,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM


@dataclass
class GenerateConfig:
    # 체크포인트 경로 (로컬 경로)
    pretrained_checkpoint: str = (
        "/home/affctiv/openvla-oft/runs/"
        "openvla-7b+ur5e_pick_and_place_dataset+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
    )

    # 모델 설정
    model_family: str = "openvla"
    use_l1_regression: bool = True       
    use_diffusion: bool = False
    use_film: bool = False
    use_proprio: bool = True            

    # 입력 설정
    num_images_in_input: int = 1         
    center_crop: bool = True             

    # 추론 설정
    unnorm_key: str = "ur5e_pick_and_place_dataset"  
    num_open_loop_steps: int = NUM_ACTIONS_CHUNK     

    # 양자화
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # LoRA 설정
    lora_rank: int = 32


def load_model(cfg: GenerateConfig):
    """VLA 모델과 action head 로드."""
    print("1. VLA 모델 로딩 중...")
    vla = get_vla(cfg)
    print(f"   llm_dim: {vla.llm_dim}")

    print("2. Processor 로딩 중...")
    processor = get_processor(cfg)

    print("3. Action head 로딩 중... (L1 regression MLP)")
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim)

    print("모델 로딩 완료!")
    return vla, processor, action_head


def setup_zmq():
    """ZMQ 소켓 설정."""
    context = zmq.Context()

    # 카메라 이미지 수신 (SUB) - 포트 5555
    cam_sock = context.socket(zmq.SUB)
    cam_sock.connect("tcp://localhost:5555")
    cam_sock.setsockopt_string(zmq.SUBSCRIBE, "")
    cam_sock.setsockopt(zmq.CONFLATE, 1)  

    # 실제 로봇 관절 상태 수신 (SUB) - 포트 5557
    state_sock = context.socket(zmq.SUB)
    state_sock.connect("tcp://localhost:5557")
    state_sock.setsockopt_string(zmq.SUBSCRIBE, "")
    state_sock.setsockopt(zmq.CONFLATE, 1)  

    # 관절 명령 전송 (PUB) - 포트 5556
    cmd_sock = context.socket(zmq.PUB)
    cmd_sock.bind("tcp://*:5556")

    print("ZMQ 소켓 설정 완료 (카메라: 5555, 상태: 5557, 명령: 5556)")
    return cam_sock, state_sock, cmd_sock


def recv_image(cam_sock) -> Optional[np.ndarray]:
    """카메라 이미지 수신."""
    try:
        image_bytes = cam_sock.recv(zmq.NOBLOCK)
        image_np = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_rgb
    except zmq.Again:
        return None


def run_inference(cfg: GenerateConfig):
    """메인 inference 루프."""

    # 모델 로드
    vla, processor, action_head = load_model(cfg)

    # ZMQ 소켓 설정
    cam_sock, state_sock, cmd_sock = setup_zmq()

    print("추론 루프 시작... (Ctrl+C로 종료)\n")

    action_queue = []
    
    # 최초 데이터 누락 방지용 초기화
    current_robot_state = np.zeros(PROPRIO_DIM, dtype=np.float32)
    
    # 시나리오 제어용 상태 변수 (0: pick, 1: place, 2: initial)
    stage = 0
    last_task_label = ""
    task = "pick object"  # 초기 태스크 지정

    while True:
        try:
            # 1. 이미지 수신
            image = recv_image(cam_sock)
            if image is None:
                time.sleep(0.01)
                continue

            # 2. 실제 로봇 관절 및 그리퍼 상태 수신 (Proprio 피드백)
            try:
                state_data = state_sock.recv_json(zmq.NOBLOCK)
                current_robot_state = np.array(state_data["joints"] + [state_data["gripper"]], dtype=np.float32)
            except zmq.Again:
                pass # 새 데이터가 아직 안왔으면 직전 상태 유지

            # [그리퍼 입력 기반 단계 전환 로직]
            actual_gripper_proprio = round(float(current_robot_state[-1]))

            if stage == 0:
                task = "pick object"
                if actual_gripper_proprio == 0.0:
                    stage = 1
            elif stage == 1:
                task = "place object"
                if actual_gripper_proprio == 1.0:
                    stage = 2
            elif stage == 2:
                task = "move to initial position"

            # 지시문 템플릿 적용
            task_label = f"In: What action should the robot take to {task}?\nOut:"

            # 지시문이 변경될 때만 로그 출력
            if task_label != last_task_label:
                print(f"📢 [태스크 전환] 스테이지: {stage} | 그리퍼 상태: {'열림(1.0)' if actual_gripper_proprio == 1.0 else '닫힘(0.0)'}")
                print(f"   새로운 지시문:\n{task_label}\n")
                last_task_label = task_label

            # action queue가 비면 모델 재쿼리
            if len(action_queue) == 0:
                obs = {
                    "full_image": image,
                    "proprio": current_robot_state,
                    "state": current_robot_state,
                }

                start = time.time()
                actions = get_vla_action(
                    cfg=cfg,
                    vla=vla,
                    processor=processor,
                    obs=obs,
                    task_label=task_label,
                    action_head=action_head,
                    proprio_projector=None,
                    noisy_action_projector=None,
                    use_film=cfg.use_film,
                )
                elapsed = time.time() - start
                
                # 🔹 [수정] 추론 완료 로그에 현재 가동 중인 태스크(task)를 직관적으로 출력하도록 변경
                print(f"✨ [추론 완료] {elapsed*1000:.1f}ms | 현재 태스크: [{task}] | chunk size: {len(actions)}")

                action_queue.extend(actions)

            # 다음 액션 실행
            action = action_queue.pop(0)

            joints = action[:6].tolist()
            gripper = float(action[6])

            print(f"관절 명령: {[f'{j:.3f}' for j in joints]} | 그리퍼: {gripper:.3f}")

            # 명령 전송 (포트 5556)
            cmd_sock.send_json({
                "joints": joints,
                "gripper": gripper,
                "raw_action": action.tolist()
            })

        except KeyboardInterrupt:
            print("\n종료합니다.")
            break
        except Exception as e:
            print(f"오류: {e}")
            continue


if __name__ == "__main__":
    cfg = GenerateConfig()
    run_inference(cfg)