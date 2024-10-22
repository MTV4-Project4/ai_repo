import cv2
import mediapipe as mp
import numpy as np

# Mediapipe 설정
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 사이드스텝 감지를 위한 기준 이동량
SIDE_STEP_THRESHOLD = 0.05  # 발목이 좌우로 움직인 최소 이동 거리

# 비디오 캡처 설정
cap = cv2.VideoCapture(0)

# 이전 프레임에서의 발목 위치 (초기값 설정)
prev_left_ankle_x = 0
prev_right_ankle_x = 0
sidestep_count = 0

# Mediapipe Pose 모델 로드
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("비디오 프레임을 읽을 수 없습니다.")
            break

        # 이미지를 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Pose 모델 적용
        results = pose.process(image)

        # 이미지를 다시 BGR로 변환
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 포즈 랜드마크 추출
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # 왼쪽 발목과 오른쪽 발목의 X 좌표 추출
            left_ankle_x = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x
            right_ankle_x = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x

            # 발목의 좌우 움직임 비교
            left_move = abs(left_ankle_x - prev_left_ankle_x)
            right_move = abs(right_ankle_x - prev_right_ankle_x)

            # 좌우 이동이 기준을 넘으면 사이드스텝으로 판정
            if left_move > SIDE_STEP_THRESHOLD or right_move > SIDE_STEP_THRESHOLD:
                sidestep_count += 1
                cv2.putText(image, f'Sidestep Detected: {sidestep_count}', (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # 현재 프레임 좌표를 이전 프레임 좌표로 업데이트
            prev_left_ankle_x = left_ankle_x
            prev_right_ankle_x = right_ankle_x

            # 포즈 랜드마크 그리기
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 기준점 그리기 (화면 중앙)
        height, width, _ = image.shape
        center_x, center_y = int(width / 2), int(height / 2)
        cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)  # 기준점
        cv2.putText(image, 'Center Point', (center_x + 10, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

        # 결과 화면에 표시
        cv2.imshow('Sidestep Detection', image)

        # 'q'를 누르면 종료
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
