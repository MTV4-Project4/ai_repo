import cv2
from mediapipe.python.solutions import pose as mp_pose
import numpy as np

# MediaPipe Pose 초기화 (한 번만 초기화)
pose = mp_pose.Pose()

# 전역 변수로 상태 및 카운트 관리
crunch_state = False  # 상체가 올라간 상태 여부
count = 0  # 크런치 카운트
y_threshold = 0.03  # 어깨와 엉덩이의 y 좌표 차이에 대한 임계값

# 자세 분석 함수 (프레임을 입력받아 크런치 동작을 분석)
def analyze_pose(frame):
    global crunch_state, count  # 전역 변수 사용
    
    # MediaPipe Pose로 자세 분석
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # 필요한 랜드마크 추출 (왼쪽 어깨, 왼쪽 엉덩이)
        left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y

        # 어깨와 엉덩이 사이의 y 좌표 차이 계산
        y_diff = left_hip_y - left_shoulder_y

        # 상체가 내려간 상태 설정 (초기 상태)
        if y_diff < y_threshold:  # 어깨가 엉덩이에 가까워짐 (상체 내려감)
            crunch_state = False

        # 상체가 올라간 상태 판정
        elif y_diff > y_threshold:  # 어깨가 엉덩이에 비해 위로 올라감 (상체 올라감)
            if not crunch_state:  # 이전에 내려간 상태였다면
                count += 1  # 크런치 카운트 증가
                crunch_state = True  # 상체가 올라간 상태로 변경

    return count