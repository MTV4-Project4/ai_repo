import cv2
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
import numpy as np

# MediaPipe Pose 초기화
pose = mp_pose.Pose()

# 전역 변수로 상태 및 카운트 관리
side_step_state = False  # 좌우로 이동한 상태 여부
count = 0  # 사이드스텝 카운트
left_last_position = 0  # 이전 왼발 위치
right_last_position = 0  # 이전 오른발 위치

# 자세 분석 함수
def analyze_pose(frame):
    global side_step_state, count, left_last_position, right_last_position  # 전역 변수 사용
    
    img = cv2.flip(frame, 1)
    
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        # 랜드마크 가져오기
        left_foot = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y,
        ]
        right_foot = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y,
        ]
        
        # 왼발과 오른발의 X 좌표 비교
        left_x = left_foot[0]
        right_x = right_foot[0]

        # 왼발 또는 오른발이 일정 거리 이상 이동했을 때 사이드스텝으로 판정
        if abs(left_x - left_last_position) > 0.05 or abs(right_x - right_last_position) > 0.05:
            side_step_state = True

        # 이전 상태가 이동 상태였고 발이 다시 중앙으로 돌아오면 카운트 증가
        if side_step_state and abs(left_x - right_x) < 0.03:
            count += 1
            side_step_state = False

        # 이전 위치 업데이트
        left_last_position = left_x
        right_last_position = right_x

    # 카운트를 반환
    return count