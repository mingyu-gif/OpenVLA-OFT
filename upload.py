import os
import shutil
from huggingface_hub import HfApi

# 1. 파일 경로 설정
CHECKPOINT_DIR = "/home/affctiv/openvla-oft/runs/openvla-7b+ur5e_pick_and_place_dataset+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
ADAPTER_DIR = os.path.join(CHECKPOINT_DIR, "lora_adapter")
STATS_PATH = os.path.join(CHECKPOINT_DIR, "dataset_statistics.json")

print("1. 통계치 파일 복사 중...")
# 2. 통계치 파일을 어댑터 폴더 안으로 복사
shutil.copy(STATS_PATH, ADAPTER_DIR)

# 3. 허깅페이스 업로드 설정
# 👇 이 부분을 본인의 허깅페이스 아이디로 꼭 변경하세요! (예: "affctiv/openvla-ur5e-2000step")
REPO_ID = "kimmingyu123/openvla-ur5e-5000step"

api = HfApi()

print(f"2. 허깅페이스에 '{REPO_ID}' 저장소 만드는 중...")
api.create_repo(repo_id=REPO_ID, exist_ok=True)

print("3. 본격적인 업로드 시작! 파일 크기에 따라 시간이 조금 걸릴 수 있습니다 🚀")
api.upload_folder(
    folder_path=ADAPTER_DIR,
    repo_id=REPO_ID,
    repo_type="model",
    ignore_patterns=["README.md"]
)
print("✅ 업로드 완료! 이제 언제 어디서든 허깅페이스에서 불러올 수 있습니다.")

