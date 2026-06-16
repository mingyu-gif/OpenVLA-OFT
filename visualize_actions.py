"""
visualize_actions.py

OpenVLA-OFT 액션 청크를 시각화합니다.
- 왼쪽: 입력 이미지 (3인칭 + 손목)
- 오른쪽: 예측된 EEF 궤적 3D 애니메이션
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import (
    get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM


# ── 모델 로드 & 추론 ──────────────────────────────────────────────────────────

cfg = GenerateConfig(
    pretrained_checkpoint="moojink/openvla-7b-oft-finetuned-libero-spatial",
    use_l1_regression=True,
    use_diffusion=False,
    use_film=False,
    num_images_in_input=2,
    use_proprio=True,
    center_crop=True,
    num_open_loop_steps=NUM_ACTIONS_CHUNK,
    unnorm_key="libero_spatial_no_noops",
)

print("모델 로드 중...")
vla = get_vla(cfg)
processor = get_processor(cfg)
action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as f:
    observation = pickle.load(f)

print("추론 중...")
actions = get_vla_action(
    cfg, vla, processor, observation,
    observation["task_description"],
    action_head, proprio_projector
)
actions = np.array(actions)  # (8, 7)


# ── EEF 궤적 계산 (delta → 누적합) ───────────────────────────────────────────

start = np.zeros(3)
trajectory = np.vstack([start, np.cumsum(actions[:, :3], axis=0)])  # (9, 3)


# ── 시각화 ────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 5))
fig.suptitle(f'Task: {observation["task_description"]}', fontsize=11)

# 왼쪽 1: 3인칭 이미지
ax_img1 = fig.add_subplot(1, 3, 1)
ax_img1.imshow(observation["full_image"])
ax_img1.set_title("Primary Camera")
ax_img1.axis("off")

# 왼쪽 2: 손목 이미지
ax_img2 = fig.add_subplot(1, 3, 2)
ax_img2.imshow(observation["wrist_image"])
ax_img2.set_title("Wrist Camera")
ax_img2.axis("off")

# 오른쪽: 3D 궤적 애니메이션
ax3d = fig.add_subplot(1, 3, 3, projection="3d")

margin = 0.05
x_range = trajectory[:, 0]
y_range = trajectory[:, 1]
z_range = trajectory[:, 2]


def init():
    ax3d.cla()
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title("Predicted EEF Trajectory")
    ax3d.set_xlim(x_range.min() - margin, x_range.max() + margin)
    ax3d.set_ylim(y_range.min() - margin, y_range.max() + margin)
    ax3d.set_zlim(z_range.min() - margin, z_range.max() + margin)
    # 전체 궤적 점선으로 미리 표시
    ax3d.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
              "gray", linestyle="--", alpha=0.3, linewidth=1)
    return []


def update(frame):
    init()
    step = frame + 1
    traj = trajectory[:step + 1]

    # 지나온 궤적
    ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], "b-o", linewidth=2, markersize=4)

    # 시작점
    ax3d.scatter(*trajectory[0], color="green", s=80, zorder=5, label="Start")

    # 현재 위치
    ax3d.scatter(*traj[-1], color="red", s=100, zorder=5, label=f"Step {step}")

    # 그리퍼 상태
    gripper = actions[frame, 6]
    gripper_str = f"{'OPEN' if gripper > 0.5 else 'CLOSE'} ({gripper:.2f})"
    ax3d.set_title(f"Step {step}/{len(actions)}  |  Gripper: {gripper_str}")
    ax3d.legend(loc="upper left", fontsize=8)
    return []


ani = animation.FuncAnimation(
    fig, update, frames=len(actions),
    init_func=init, interval=600, repeat=True
)

plt.tight_layout()
plt.show()
