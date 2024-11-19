import cv2
import numpy as np
import time
from ultralytics import YOLO
from mediapipe.python.solutions import pose as mp_pose
import torch

# YOLOv8 모델 불러오기
#model = YOLO("C:/MTV4/YOLO/yolov8m.pt")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = YOLO("C:/MTV4/YOLO/yolov8l.pt")  # GPU 사용
model.to(device)

# MediaPipe Pose 초기화
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class KickCounter:
    def __init__(self):
        self.kick_count = 0
        self.ball_in_contact = False
        self.last_kick_time = 0
        self.previous_ball_position = None
        self.consecutive_frames_without_ball = 0
        self.contact_frames = 0

    def reset(self):
        self.kick_count = 0
        self.ball_in_contact = False
        self.last_kick_time = 0
        self.previous_ball_position = None
        self.consecutive_frames_without_ball = 0
        self.contact_frames = 0

# 전역 카운터 객체 생성
kick_counter = KickCounter()

def reset_counter():
    kick_counter.reset()

def detect_kick(ball_position, left_foot_position, right_foot_position, threshold=100):  # 임계값 조정(65->85으로 수정)
    if ball_position is None:
        return False
        
    if left_foot_position is None and right_foot_position is None:
        return False

    # 왼발 접촉 검사
    left_contact = False
    if left_foot_position is not None:
        distance_left = np.linalg.norm(ball_position - left_foot_position)
        if distance_left < threshold:
            left_contact = True
            print(f"왼발 접촉: {distance_left:.2f}")

    # 오른발 접촉 검사
    right_contact = False
    if right_foot_position is not None:
        distance_right = np.linalg.norm(ball_position - right_foot_position)
        if distance_right < threshold:
            right_contact = True
            print(f"오른발 접촉: {distance_right:.2f}")

    return left_contact or right_contact

def analyze_kick(frame):
    # YOLO로 공 감지
    results = model(frame, conf=0.05)  # 신뢰도 임계값 유지
    ball_position = None
    max_confidence = 0
    
    if results[0].boxes:
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            if class_id == 32 and confidence > max_confidence:  # sports ball
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width = x2 - x1
                height = y2 - y1
                
                # 종횡비와 크기 조건
                aspect_ratio = width / height if height != 0 else 0
                if 0.7 < aspect_ratio < 1.3 and 5 < width < 150:
                    ball_position = np.array([(x1 + x2) // 2, (y1 + y2) // 2])
                    max_confidence = confidence
                    
                    # 시각화
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Ball {confidence:.2f}", (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # MediaPipe로 발 위치 감지
    pose_results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    left_foot_position = None
    right_foot_position = None

    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        # 발 위치 계산 (발끝만 사용)
        left_foot = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

        if left_foot.visibility > 0.5:
            left_foot_position = np.array([
                int(left_foot.x * frame.shape[1]),
                int(left_foot.y * frame.shape[0])
            ])

        if right_foot.visibility > 0.5:
            right_foot_position = np.array([
                int(right_foot.x * frame.shape[1]),
                int(right_foot.y * frame.shape[0])
            ])

    current_time = time.time()

    # 킥 감지 로직
    is_contact = False
    if ball_position is not None:
        is_contact = detect_kick(ball_position, left_foot_position, right_foot_position)
        
        if is_contact and not kick_counter.ball_in_contact:
            if (current_time - kick_counter.last_kick_time) > 0.15:  # 시간 간격 조정
                kick_counter.kick_count += 1
                kick_counter.ball_in_contact = True
                kick_counter.last_kick_time = current_time
                print(f"킥 감지! 현재 킥 횟수: {kick_counter.kick_count}")
        elif not is_contact and (current_time - kick_counter.last_kick_time) > 0.1:
            kick_counter.ball_in_contact = False

    # 시각화
    if ball_position is not None:
        color = (0, 0, 255) if is_contact else (0, 255, 0)
        cv2.circle(frame, tuple(ball_position), 10, color, -1)
        cv2.putText(frame, "Ball", tuple(ball_position), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if left_foot_position is not None:
        cv2.circle(frame, tuple(left_foot_position), 10, (255, 0, 0), -1)
        cv2.putText(frame, "L", tuple(left_foot_position), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if right_foot_position is not None:
        cv2.circle(frame, tuple(right_foot_position), 10, (0, 0, 255), -1)
        cv2.putText(frame, "R", tuple(right_foot_position), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 현재 상태 표시
    status = "Contact!" if is_contact else "No Contact"
    cv2.putText(frame, f"Kicks: {kick_counter.kick_count} - {status}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # 거리 정보 표시
    if ball_position is not None:
        if left_foot_position is not None:
            distance_left = np.linalg.norm(ball_position - left_foot_position)
            cv2.putText(frame, f"L:{distance_left:.1f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if right_foot_position is not None:
            distance_right = np.linalg.norm(ball_position - right_foot_position)
            cv2.putText(frame, f"R:{distance_right:.1f}", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Kick Detection", frame)
    cv2.waitKey(1)

    return kick_counter.kick_count