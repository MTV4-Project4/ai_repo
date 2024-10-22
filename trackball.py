import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8 모델 불러오기
model = YOLO("yolov8n.pt")  # YOLOv8n 모델을 사용합니다. 필요에 따라 yolov8s.pt 등 다른 모델을 선택할 수 있습니다.

# 비디오 캡처 설정
cap = cv2.VideoCapture(0)  # 웹캠으로 실시간 영상 받기
kick_count = 0  # 킥 카운터
previous_position = None  # 이전 프레임에서의 공 위치

# 킥 감지 함수
def detect_kick(current_position, previous_position, threshold=20):
    if previous_position is None:
        return False
    
    distance = np.linalg.norm(np.array(current_position) - np.array(previous_position))
    if distance > threshold:  # 공의 위치 변화가 threshold를 넘을 때 킥으로 판정
        return True
    return False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8을 사용하여 객체 감지
    results = model(frame)
    ball_position = None

    # YOLOv8 결과에서 'sports ball'을 찾음
    for result in results.xyxy[0]:
        class_id = int(result[5])  # 객체 클래스 ID
        if class_id == 32:  # 'sports ball' 클래스는 32번입니다.
            x1, y1, x2, y2 = map(int, result[:4])  # 경계 상자 좌표
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            ball_position = (center_x, center_y)
            break

    if ball_position is not None and previous_position is not None:
        if detect_kick(ball_position, previous_position):
            kick_count += 1
            print(f"킥 횟수: {kick_count}")

    previous_position = ball_position  # 현재 위치를 이전 위치로 저장

    # 화면에 공의 위치 표시
    if ball_position is not None:
        cv2.circle(frame, ball_position, 10, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()