import cv2
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
import numpy as np

# MediaPipe Pose 초기화
pose = mp_pose.Pose()

# 전역 변수로 상태 및 카운트 관리
crunch_state = False  # 몸이 위로 올라간 상태 여부
count = 0  # 크런치 카운트

# 3 점을 가지고 각도를 구하는 함수
def calculate_angle(a, b, c):  # b가 가운데
    a = np.array(a[:2])  # x, y 좌표만 사용
    b = np.array(b[:2])
    c = np.array(c[:2])
    
    ba = a - b
    bc = c - b
    
    cosangle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))  # 내적 / 길이*길이
    arccos = np.arccos(cosangle)
    degree = np.degrees(arccos)  # 각도로 바꿈
    return degree

# 자세 분석 함수
def analyze_pose(frame):
    global crunch_state, count  # 전역 변수 사용
    
    img = cv2.flip(frame, 1)
    
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        left_shoulder = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x, 
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
        ]
        left_hip = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y,
        ]
        left_knee = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y,
        ]
        left_angle = calculate_angle(left_shoulder, left_hip, left_knee)

        right_shoulder = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, 
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
        ]
        right_hip = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y,
        ]
        right_knee = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y,
        ]
        right_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        
        # 크런치 동작 판정
        if left_angle <= 45 and right_angle <= 45:
            crunch_state = True  # 몸이 올라간 상태로 판정
        if left_angle >= 90 and right_angle >= 90:
            if crunch_state:  # 이전 상태가 올라간 상태였다면
                count += 1  # 크런치 카운트 증가
                crunch_state = False  # 내려간 상태로 변경

    # 카운트를 반환
    return count