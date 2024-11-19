import cv2
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
import numpy as np
import time

pose = mp_pose.Pose()

# 전역 변수로 상태 및 카운트 관리
left_lifted = False  # 왼발 들기 상태
right_lifted = False  # 오른발 들기 상태
count = 0  # 전체 카운트
last_left_lift_time = 0  # 마지막 왼발 리프팅 시간
last_right_lift_time = 0  # 마지막 오른발 리프팅 시간
min_lift_interval = 0.2  # 최소 리프팅 간격 (초 단위)
angle_threshold = 80  # 다리 들기 각도 임계값
height_threshold = 0.05  # 발목 높이 임계값 (0~1 스케일, 정면에서 다리의 높이를 확인하기 위한 임계값)

# 초기 상태 설정 변수
initialized = False  # 초기화 여부
initial_left_angle = None  # 초기 왼발 각도
initial_right_angle = None  # 초기 오른발 각도

# 3 점을 가지고 각도를 구하는 함수
def calculate_angle(a, b, c):  # b가 가운데
    a = np.array(a[:2])  # x, y 좌표만 사용
    b = np.array(b[:2])
    c = np.array(c[:2])
    
    ba = a - b
    bc = c - b
    
    cosangle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))  # 내적 / 길이*길이
    arccos = np.arccos(cosangle)
    degree = np.degrees(arccos)  # 각도로 변환
    return degree

# 자세 분석 함수 (제자리 뛰기)
def analyze_jump(frame):
    global left_lifted, right_lifted, count, last_left_lift_time, last_right_lift_time
    global initialized, initial_left_angle, initial_right_angle  # 초기화 관련 변수

    debug_frame = frame.copy()  # 디버깅용 이미지 생성
    
    # Mediapipe로 이미지 처리
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    current_time = time.time()  # 현재 시간
    
    if results.pose_landmarks:
        # Mediapipe 랜드마크 및 연결 그리기
        mp_drawing.draw_landmarks(
            debug_frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
        )
        
        # 왼쪽 다리 각도 및 발목 높이 계산
        left_hip = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
        ]
        left_knee = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y
        ]
        left_ankle = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y
        ]
        left_angle = calculate_angle(left_hip, left_knee, left_ankle)
        left_ankle_height = left_ankle[1]  # 발목의 y좌표 (세로 높이)

        # 오른쪽 다리 각도 및 발목 높이 계산
        right_hip = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y
        ]
        right_knee = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y
        ]
        right_ankle = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y
        ]
        right_angle = calculate_angle(right_hip, right_knee, right_ankle)
        right_ankle_height = right_ankle[1]  # 발목의 y좌표 (세로 높이)

        # 초기화 단계: 각도가 임계값 내에 있는 경우에만 초기 각도를 설정
        if not initialized and left_angle < angle_threshold and right_angle < angle_threshold:
            initial_left_angle = left_angle
            initial_right_angle = right_angle
            initialized = True
            print("초기화 완료: 초기 각도 저장")
            return count  # 초기화 이후 카운트 시작

        # 왼발이 임계값 이상 들렸을 때
        if (left_angle >= angle_threshold or left_ankle_height < height_threshold) and not left_lifted and (current_time - last_left_lift_time) > min_lift_interval:
            count += 1
            left_lifted = True  # 왼발이 올려진 상태로 변경
            last_left_lift_time = current_time
            print(f"왼발 들기 카운트: {count}")
        elif left_angle < angle_threshold and left_ankle_height >= height_threshold:
            left_lifted = False  # 왼발이 내려가면 상태 초기화

        # 오른발이 임계값 이상 들렸을 때
        if (right_angle >= angle_threshold or right_ankle_height < height_threshold) and not right_lifted and (current_time - last_right_lift_time) > min_lift_interval:
            count += 1
            right_lifted = True  # 오른발이 올려진 상태로 변경
            last_right_lift_time = current_time
            print(f"오른발 들기 카운트: {count}")
        elif right_angle < angle_threshold and right_ankle_height >= height_threshold:
            right_lifted = False  # 오른발이 내려가면 상태 초기화

        # 디버깅 정보 화면에 표시
        cv2.putText(debug_frame, f"Jump Count: {count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(debug_frame, f"Left Angle: {left_angle:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(debug_frame, f"Right Angle: {right_angle:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        cv2.putText(debug_frame, "No Landmarks Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 디버깅 화면 표시
    cv2.imshow("Debug View", debug_frame)
    cv2.waitKey(1)

    return count
