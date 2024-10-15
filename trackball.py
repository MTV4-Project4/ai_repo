import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)

# 비디오 캡처 초기화
cap = cv2.VideoCapture(0)  # 웹캠 사용

# 차기 횟수 초기화 및 상태 변수 설정
kick_count = 0
in_kick_position = False

def calculate_angle(a, b, c):
    """세 점의 각도를 계산하는 함수"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def detect_kick(landmarks):
    """공을 차는 동작인지 확인하는 함수"""
    # 필요한 랜드마크 좌표 추출 (예: 발목과 무릎)
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    
    # 무릎 각도 계산
    knee_angle = calculate_angle(hip, knee, ankle)
    
    # 공 차기 동작 판단 (예: 무릎 각도가 특정 각도 이하일 때)
    return knee_angle < 140

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # BGR 이미지를 RGB로 변환
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # MediaPipe Pose 처리
    results = pose.process(image)
    
    # 랜드마크가 감지되면 공 차기 확인
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        if detect_kick(landmarks):
            if not in_kick_position:
                kick_count += 1
                in_kick_position = True
        else:
            in_kick_position = False
    
    # 결과 출력
    cv2.putText(frame, f'Kicks: {kick_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow('Kick Counter', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()