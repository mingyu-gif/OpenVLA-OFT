"""
test_ur5e_inference.py

이미지 한 장을 입력으로 받아 관절 각도를 출력하는 테스트 코드.
"""

import json
import sys
import numpy as np
import torch
from dataclasses import dataclass
from PIL import Image

sys.path.append("/home/affctiv/openvla-oft")

from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM
from experiments.robot.openvla_utils import (
    get_vla_action,
    load_component_state_dict,
    find_checkpoint_file,
)

DEVICE = torch.device("cuda:0")

@dataclass
class GenerateConfig:
    pretrained_checkpoint: str = (
        "/home/affctiv/openvla-oft/runs/"
        "openvla-7b+ur5e_pick_and_place_dataset+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--5000_chkpt"
    )
    model_family: str = "openvla"
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    use_proprio: bool = False
    num_images_in_input: int = 1
    center_crop: bool = True
    unnorm_key: str = "ur5e_pick_and_place_dataset"
    num_open_loop_steps: int = NUM_ACTIONS_CHUNK
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    lora_rank: int = 32


cfg = GenerateConfig()

# 1. HF Auto Classes 등록
AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

# 2. 모델 직접 로드 (safetensors 포맷)
print("1. VLA 모델 로딩 중...")
vla = AutoModelForVision2Seq.from_pretrained(
    cfg.pretrained_checkpoint,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to(DEVICE)
vla.eval()

# 3. dataset_statistics 로드
print("2. 정규화 통계 로딩 중...")
import os
stats_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
with open(stats_path, "r") as f:
    vla.norm_stats = json.load(f)

# 4. Processor 로드
print("3. Processor 로딩 중...")
processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)

# 5. Action head 로드
print("4. Action head 로딩 중...")
action_head = L1RegressionActionHead(
    input_dim=vla.llm_dim,
    hidden_dim=vla.llm_dim,
    action_dim=ACTION_DIM,
).to(torch.bfloat16).to(DEVICE)
action_head.eval()

checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "action_head")
state_dict = load_component_state_dict(checkpoint_path)
action_head.load_state_dict(state_dict)
print("모델 로딩 완료!")

# 6. 이미지 로드
image_path = "/home/affctiv/openvla-oft/runs/test_Color.png"
image = np.array(Image.open(image_path).convert("RGB"))

obs = {"full_image": image}
task_label = "pick and place object"

print(f"\n태스크: {task_label}")
print("추론 중...")

actions = get_vla_action(
    cfg=cfg,
    vla=vla,
    processor=processor,
    obs=obs,
    task_label=task_label,
    action_head=action_head,
)

print(f"\n=== 추론 결과 (action chunk: {len(actions)}개) ===")
for i, action in enumerate(actions):
    joints = action[:6].tolist()
    gripper = float(action[6])
    print(f"[{i+1}] 관절: {[f'{j:.4f}' for j in joints]} | 그리퍼: {gripper:.4f}")