import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from huggingface_hub import hf_hub_download

BASE_MODEL = "openvla/openvla-7b"
MY_LORA_REPO = "kimmingyu123/openvla-ur5e-12000step" 

print("1. 뼈대 모델 및 프로세서 불러오는 중... ☕")
processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
base_vla = AutoModelForVision2Seq.from_pretrained(
    BASE_MODEL, 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
)

print("2. 내 똑똑한 칩(LoRA 어댑터) 허깅페이스에서 다운로드 및 결합 중...")
vla = PeftModel.from_pretrained(base_vla, MY_LORA_REPO)
vla = vla.merge_and_unload().to("cuda")
vla.eval()

print("3. 통계치 파일 다운로드 및 모델에 주입 중...")
stats_file_path = hf_hub_download(repo_id=MY_LORA_REPO, filename="dataset_statistics.json")
with open(stats_file_path, "r") as f:
    dataset_stats = json.load(f)

# 🚨 궁극의 해결책: JSON 파일 안에 통계가 얼마나 깊이 숨어있든 끝까지 찾아내는 탐지기 함수
def extract_q_vals(data):
    if isinstance(data, dict):
        if "q01" in data: return data["q01"], data.get("q99")
        if "min" in data: return data["min"], data.get("max")
        for key, val in data.items():
            res = extract_q_vals(val)
            if res is not None:
                return res
    return None

# 탐지기 가동!
q01_vals, q99_vals = extract_q_vals(dataset_stats)

# 찾은 값을 모델이 가장 좋아하는 형태로 강제 주입
vla.norm_stats["my_custom_ur5e"] = {
    "action": {
        "q01": q01_vals,
        "q99": q99_vals
    }
}

print("4. 테스트 준비 중...")
image_path = "/home/affctiv/openvla-oft/runs/test_Color.png" 
image = Image.open(image_path).convert("RGB")

instruction = "pick up the block" 
prompt = f"In: What action should the robot take to {instruction}?\nOut:"

print(f"\n👉 입력된 명령어: {instruction}")
print("5. 추론 시작! 로봇이 사진을 보고 계산 중입니다...")

inputs = processor(prompt, image).to("cuda", dtype=torch.bfloat16)

with torch.inference_mode():
    real_action = vla.predict_action(**inputs, unnorm_key="my_custom_ur5e")

print("\n🎉🎉🎉 추론 결과 🎉🎉🎉")
print(f"🤖 실제 관절 제어 명령 (XYZ, Roll-Pitch-Yaw, Gripper): \n{real_action}")
print("=======================")

