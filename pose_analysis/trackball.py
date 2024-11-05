import cv2
import numpy as np
from ultralytics import YOLO
from mediapipe.python.solutions import pose as mp_pose

# YOLOv8 모델 불러오기
model = YOLO("C:/MTV4/YOLO/yolov8n.pt")

# MediaPipe Pose 초기화
pose = mp_pose.Pose()

# 킥 카운터와 이전 공 위치 초기화
kick_count = 0
previous_ball_position = None

# 공과 발의 접촉 감지 함수
def detect_kick(ball_position, left_foot_position, right_foot_position, threshold=20):
    if not ball_position:
        return False

    # 공과 왼발의 거리 계산
    if left_foot_position:
        left_distance = np.linalg.norm(np.array(ball_position) - np.array(left_foot_position))
        if left_distance < threshold:
            return True

    # 공과 오른발의 거리 계산
    if right_foot_position:
        right_distance = np.linalg.norm(np.array(ball_position) - np.array(right_foot_position))
        if right_distance < threshold:
            return True

    return False

# 이미지 분석 함수
def analyze_kick(frame):
    global kick_count, previous_ball_position

    # YOLO를 사용하여 공 탐지
    results = model(frame)
    ball_position = None

    for result in results.xyxy[0]:
        class_id = int(result[5])  # 객체 클래스 ID
        if class_id == 32:  # 'sports ball' 클래스 ID
            x1, y1, x2, y2 = map(int, result[:4])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            ball_position = (center_x, center_y)
            break  # 첫 번째 공만 탐지

    # MediaPipe Pose를 사용하여 발 위치 탐지
    left_foot_position = None
    right_foot_position = None
    pose_results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if pose_results.pose_landmarks:
        left_foot = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        right_foot = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
        left_foot_position = (int(left_foot.x * frame.shape[1]), int(left_foot.y * frame.shape[0]))
        right_foot_position = (int(right_foot.x * frame.shape[1]), int(right_foot.y * frame.shape[0]))

    # 공과 발의 접촉 판정
    if ball_position and detect_kick(ball_position, left_foot_position, right_foot_position):
        kick_count += 1
        print(f"킥 횟수: {kick_count}")

    # 이전 공 위치 업데이트
    previous_ball_position = ball_position

    # 킥 카운트 반환
    return kick_count
