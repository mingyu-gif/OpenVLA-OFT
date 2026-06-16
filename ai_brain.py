import torch, cv2, zmq, json
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from huggingface_hub import hf_hub_download

def extract_q_vals(data):
    if isinstance(data, dict):
        if "q01" in data: return data["q01"], data.get("q99")
        if "min" in data: return data["min"], data.get("max")
        for key, val in data.items():
            res = extract_q_vals(val)
            if res is not None: return res
    return None

def main():
    print("🧠 뇌 가동 준비 중... (GPU 로딩)")

    REPO = "kimmingyu123/openvla-ur5e-500step"
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    vla = PeftModel.from_pretrained(
        AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ), REPO
    ).merge_and_unload().to("cuda")
    vla.eval()

    stats = json.load(open(hf_hub_download(repo_id=REPO, filename="dataset_statistics.json"), "r"))
    result = extract_q_vals(stats)
    assert result is not None, "❌ dataset_statistics.json 파싱 실패"
    q01, q99 = result
    vla.norm_stats["my_custom_ur5e"] = {"action": {"q01": q01, "q99": q99}}

    print("🧠 모델 세팅 완료! 통신 채널을 엽니다.")

    context = zmq.Context()
    cam_sock = context.socket(zmq.SUB)
    cam_sock.connect("tcp://localhost:5555")
    cam_sock.setsockopt_string(zmq.SUBSCRIBE, "")
    cam_sock.setsockopt(zmq.CONFLATE, 1)

    cmd_sock = context.socket(zmq.PUB)
    cmd_sock.bind("tcp://*:5556")

    # ─── 상태머신 ───────────────────────────────
    PICK_PROMPT  = "In: What action should the robot take to pick up the block?\nOut:"
    PLACE_PROMPT = "In: What action should the robot take to place the block?\nOut:"

    state = "PICK"   # 초기 상태
    GRIPPER_CLOSE_THRESHOLD = 0.5  # 이 값 이상이면 "닫힘"으로 판단
    # ────────────────────────────────────────────

    print("📸 추론 루프 가동! 초기 상태: PICK")

    while True:
        try:
            # 최신 프레임 수신
            image_bytes = cam_sock.recv()
            image_pil = Image.fromarray(
                cv2.cvtColor(
                    cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB
                )
            )

            # 상태에 따라 프롬프트 선택
            prompt = PICK_PROMPT if state == "PICK" else PLACE_PROMPT

            # 추론
            inputs = processor(
                text=prompt,
                images=image_pil,
                return_tensors="pt"
            ).to("cuda", dtype=torch.bfloat16)

            with torch.inference_mode():
                action = vla.predict_action(**inputs, unnorm_key="my_custom_ur5e")

            target_angles = action[:6].tolist()
            print(f"🎯 예측된 관절 각도: {target_angles}")
            gripper_cmd   = float(action[6])  # 0.0(열림) ~ 1.0(닫힘)

            # 상태 전환 감지
            if state == "PICK" and gripper_cmd >= GRIPPER_CLOSE_THRESHOLD:
                print(f"🤏 그리퍼 닫힘 감지 ({gripper_cmd:.2f}) → PLACE 모드로 전환!")
                state = "PLACE"
            elif state == "PLACE" and gripper_cmd < GRIPPER_CLOSE_THRESHOLD:
                print(f"✋ 그리퍼 열림 감지 ({gripper_cmd:.2f}) → PICK 모드로 전환!")
                state = "PICK"

            print(f"[{state}] 🎯 관절: {[f'{a:.3f}' for a in target_angles]} | 그리퍼: {gripper_cmd:.2f}")

            # 명령 전송 (관절 6개 + 그리퍼 1개)
            cmd_sock.send_json({
                "joints":  target_angles,
                "gripper": gripper_cmd,
                "state":   state
            })

        except zmq.ZMQError as e:
            print(f"⚠️ ZMQ 오류: {e}, 재시도...")
            continue
        except Exception as e:
            print(f"❌ 추론 오류: {e}")
            continue

if __name__ == '__main__':
    main()