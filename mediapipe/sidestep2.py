import cv2
import mediapipe as mp
import numpy as np

# Mediapipe 설정
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 사이드스텝 감지할 임계값 (x축 이동에 따른 차이값)
SIDE_STEP_THRESHOLD = 0.05

# 사이드스텝 카운터 변수
side_step_count = 0

# 비디오 캡처
cap = cv2.VideoCapture(0)

# Mediapipe 모델 로드
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    prev_left_hip_x = None
    prev_right_hip_x = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("비디오 프레임을 읽을 수 없습니다.")
            break

        # 이미지를 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Mediapipe 모델 적용 (Pose)
        results = pose.process(image)

        # 이미지를 다시 BGR로 변환
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image_height, image_width, _ = image.shape

        # 사이드스텝 상태 확인
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # 왼쪽 엉덩이, 오른쪽 엉덩이, 왼쪽 발목, 오른쪽 발목의 x 좌표 추출
            left_hip_x = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x
            right_hip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x

            if prev_left_hip_x is not None and prev_right_hip_x is not None:
                # 왼쪽 또는 오른쪽으로 이동했는지 판정 (현재 좌표와 이전 좌표 비교)
                if abs(left_hip_x - prev_left_hip_x) > SIDE_STEP_THRESHOLD:
                    step_direction = 'Left Side Step' if left_hip_x < prev_left_hip_x else 'Right Side Step'
                    color = (0, 255, 0)  # 연두색

                    # 사이드스텝이 발생할 때마다 카운터 증가
                    side_step_count += 1

                elif abs(right_hip_x - prev_right_hip_x) > SIDE_STEP_THRESHOLD:
                    step_direction = 'Right Side Step' if right_hip_x > prev_right_hip_x else 'Left Side Step'
                    color = (0, 255, 0)  # 연두색

                    # 사이드스텝이 발생할 때마다 카운터 증가
                    side_step_count += 1

                else:
                    step_direction = 'No Side Step'
                    color = (0, 0, 255)  # 빨간색
            else:
                step_direction = 'No Previous Data'
                color = (255, 255, 0)  # 노란색 (데이터 없음)

            # 이전 좌표 업데이트
            prev_left_hip_x = left_hip_x
            prev_right_hip_x = right_hip_x

            # 화면에 사이드스텝 상태 표시
            cv2.putText(image, step_direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # 사이드스텝 카운트 표시
            cv2.putText(image, f'Side Steps: {side_step_count}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            # 포즈 랜드마크 그리기
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS
            )

        # 화면 출력
        cv2.imshow('Side Step Detection & Counter', image)

        # 'q'를 누르면 종료
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
