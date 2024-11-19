import cv2
from mediapipe.python.solutions import pose as mp_pose
import mediapipe as mp

# MediaPipe Pose 초기화 (한 번만 초기화)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 전역 변수로 상태 및 카운트 관리
crunch_state = False  
count = 0  
y_threshold = 0.02  

def analyze_pose(frame):
    global crunch_state, count  
    
    # 디버깅을 위한 프레임 복사
    debug_frame = frame.copy()

    # RGB로 변환하여 처리
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # 기본값 설정
    y_diff = 0.0  # 기본값을 설정하여 UnboundLocalError 방지

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # 어깨와 엉덩이의 Y 좌표 계산
        left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
        y_diff = left_hip_y - left_shoulder_y

        # 크런치 상태 판정
        if y_diff < y_threshold:  
            crunch_state = False
        elif y_diff > y_threshold:  
            if not crunch_state:  
                count += 1  
                crunch_state = True  

        # 랜드마크 그리기 (시각화)
        mp_drawing = mp.solutions.drawing_utils  
        mp_drawing.draw_landmarks(debug_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    else:
        # 랜드마크가 없는 경우 디버깅 메시지 출력
        cv2.putText(debug_frame, "No Landmarks Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 디버깅 프레임에 카운트 및 y_diff 출력
    cv2.putText(debug_frame, f"Crunch Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(debug_frame, f"y_diff: {y_diff:.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 결과 이미지 보여주기 (Received Frame과 Debug Frame만 표시)
    cv2.imshow('Debug Frame', debug_frame)  # 디버깅 화면
    cv2.imshow('Received Frame', frame)     # Received Frame (원본 입력)
    cv2.waitKey(1)  # 프레임을 보기 위해 짧게 대기

    # 함수가 카운트만 반환
    return count
