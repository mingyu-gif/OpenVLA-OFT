import pyrealsense2 as rs
import zmq
import cv2
import numpy as np

def main():
    # 1. ZMQ 방송국(Publisher) 설립
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5555") # 👈 5555번 포트로 전 세계(로컬 네트워크)에 방송 시작!

    # 2. RealSense 카메라 세팅
    pipeline = rs.pipeline()
    config = rs.config()
    # 해상도 640x480, 초당 30프레임으로 컬러 사진 세팅
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    pipeline.start(config)
    print("🎥 리얼센스 카메라 방송 시작! (포트 5555에서 대기 중...)")

    try:
        while True:
            # 3. 찰칵! 0.03초마다 프레임 가져오기
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # 4. 사진을 컴퓨터가 읽기 편한 배열(Numpy)로 변환
            color_image = np.asanyarray(color_frame.get_data())

            # 5. 인터넷으로 쏘기 위해 가볍게 압축(.jpg)
            _, encoded_image = cv2.imencode('.jpg', color_image)
            
            # 6. ZMQ로 빛의 속도로 발송! 🚀
            socket.send(encoded_image.tobytes())

            # (선택) 내 화면에도 잘 찍히는지 띄워보기
            cv2.imshow('RealSense Camera', color_image)
            if cv2.waitKey(1) == ord('q'): # 'q' 누르면 종료
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
