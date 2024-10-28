import cv2
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
import numpy as np

pose = mp_pose.Pose()

# 전역 변수로 상태 및 카운트 관리
squat_state = False  # down 상태 여부
count = 0  # 스쿼트 카운트

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
    global squat_state, count  # 전역 변수 사용
    
    img = cv2.flip(frame, 1)
    
    # Mediapipe를 사용하여 이미지에서 포즈 추출
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # 포즈 랜드마크가 감지된 경우에만 작업 진행
    if results.pose_landmarks:
        # 사람의 팔과 다리 랜드마크를 먼저 인식
        landmarks = results.pose_landmarks.landmark
        
        # 팔과 다리의 주요 랜드마크 인식 여부 확인
        key_points = [
            mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE
        ]
        
        detected = all(landmarks[point].visibility > 0.5 for point in key_points)
        
        # 팔과 다리의 주요 랜드마크가 인식된 경우에만 스쿼트 분석 수행
        if detected:
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
            )
            
            # 좌표 추출 및 각도 계산
            left_hip = [
                landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, 
                landmarks[mp_pose.PoseLandmark.LEFT_HIP].y,
            ]
            left_knee = [
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y,
            ]
            left_ankle = [
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y,
            ]
            left_angle = calculate_angle(left_hip, left_knee, left_ankle)

            right_hip = [
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, 
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y, 
            ]
            right_knee = [
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y,
            ]
            right_ankle = [
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y,
            ]
            right_angle = calculate_angle(right_hip, right_knee, right_ankle)
            
            # down 임계값을 90도로 변경하고 up 임계값을 120도로 변경
            if left_angle <= 90 and right_angle <= 90:
                cv2.putText(img, "down", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)
                squat_state = True  # down 상태
            elif left_angle >= 120 and right_angle >= 120:
                cv2.putText(img, "up", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)
                if squat_state:  # 이전 상태가 down이었다면
                    count += 1  # 스쿼트 카운트 증가
                    squat_state = False  # up 상태로 변경

    # img 대신 count를 리턴
    return count
