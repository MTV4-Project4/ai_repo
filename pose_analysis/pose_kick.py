import cv2
import numpy as np
import time
from ultralytics import YOLO
from mediapipe.python.solutions import pose as mp_pose

# YOLOv8 모델 불러오기
model = YOLO("C:/MTV4/YOLO/yolov8n.pt")

# MediaPipe Pose 초기화
pose = mp_pose.Pose()

# 킥 카운터와 이전 공 위치 초기화
kick_count = 0
previous_ball_position = None
ball_in_contact = False  # 공과 발이 닿아있는 상태인지 여부
last_kick_time = 0  # 마지막 킥이 발생한 시간
min_kick_interval = 0.5  # 최소 킥 간격 (초)

# 공과 발의 접촉 감지 함수
def detect_kick(ball_position, left_foot_position, right_foot_position, threshold=30):
    if ball_position is None:
        return False
    if left_foot_position is not None and np.linalg.norm(ball_position - left_foot_position) < threshold:
        return True
    if right_foot_position is not None and np.linalg.norm(ball_position - right_foot_position) < threshold:
        return True
    return False

# 이미지 분석 함수
def analyze_kick(frame):
    global kick_count, previous_ball_position, ball_in_contact, last_kick_time

    # YOLO를 사용하여 공 탐지
    results = model(frame)
    ball_position = None

    # YOLOv8 결과에서 'sports ball' 또는 'person'만 탐지
    if results[0].boxes:
        for box in results[0].boxes:
            class_id = int(box.cls[0])  # 객체 클래스 ID
            
            # 'sports ball'과 'person'만 탐지 (sports ball: ID 32, person: ID 0)
            if class_id == 32 or class_id == 0:
                if class_id == 32:  # 공 탐지
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    ball_position = np.array([center_x, center_y])

                    # 공의 크기 조건 추가 (너무 크거나 작은 물체를 제외)
                    width, height = x2 - x1, y2 - y1
                    if width < 10 or height < 10:
                        ball_position = None
                    elif previous_ball_position is not None:
                        # 이전 위치와의 거리를 기준으로 유효성 검사
                        movement = np.linalg.norm(ball_position - previous_ball_position)
                        if movement > 100:
                            ball_position = None
                    break

    # MediaPipe Pose를 사용하여 발 위치 탐지
    left_foot_position = None
    right_foot_position = None
    pose_results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if pose_results and pose_results.pose_landmarks:
        left_foot = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        right_foot = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
        left_foot_position = np.array([int(left_foot.x * frame.shape[1]), int(left_foot.y * frame.shape[0])])
        right_foot_position = np.array([int(right_foot.x * frame.shape[1]), int(right_foot.y * frame.shape[0])])

    # 현재 시간 가져오기
    current_time = time.time()

    # 공과 발의 접촉 판정
    if ball_position is not None and detect_kick(ball_position, left_foot_position, right_foot_position):
        if not ball_in_contact and (current_time - last_kick_time) > min_kick_interval:
            kick_count += 1
            ball_in_contact = True  # 공과 발이 닿아있음
            last_kick_time = current_time  # 킥이 발생한 시간 업데이트
            print(f"킥 횟수: {kick_count}")
    elif ball_position is None or not detect_kick(ball_position, left_foot_position, right_foot_position):
        # 공이 발에서 떨어졌음을 확인하면 접촉 상태 해제
        ball_in_contact = False

    # 이전 공 위치 업데이트
    previous_ball_position = ball_position

    # 시각화
    if ball_position is not None:
        cv2.circle(frame, tuple(ball_position), 10, (0, 255, 0), -1)  # 공 위치 표시 (초록색 원)
    if left_foot_position is not None:
        cv2.circle(frame, tuple(left_foot_position), 10, (255, 0, 0), -1)  # 왼발 위치 표시 (파란색 원)
    if right_foot_position is not None:
        cv2.circle(frame, tuple(right_foot_position), 10, (0, 0, 255), -1)  # 오른발 위치 표시 (빨간색 원)

    # 킥 횟수 표시
    cv2.putText(frame, f"Kicks: {kick_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # 화면에 출력하여 시각적으로 확인
    cv2.imshow("Kick Detection", frame)
    cv2.waitKey(1)

    # 킥 카운트 반환
    return kick_count













