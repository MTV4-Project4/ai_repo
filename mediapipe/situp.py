import cv2
import mediapipe as mp
import numpy as np

# Mediapipe 설정
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """세 점(a, b, c)의 각도를 계산"""
    a = np.array(a)  # 첫 번째 점
    b = np.array(b)  # 중간 점
    c = np.array(c)  # 세 번째 점

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# 싯업을 감지하기 위한 허리, 엉덩이, 무릎 각도 기준
MIN_ANGLE = 70  # 몸이 완전히 누워있을 때 허리와 엉덩이 각도
MAX_ANGLE = 140  # 상체를 일으켰을 때 허리와 엉덩이 각도

# 비디오 캡처
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("비디오 프레임을 읽을 수 없습니다.")
        break

    # 이미지를 RGB로 변환
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Mediapipe 포즈 모델 적용
    results = pose.process(image)

    # 이미지를 다시 BGR로 변환
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        # 포즈 랜드마크 추출
        landmarks = results.pose_landmarks.landmark

        # 필요한 랜드마크 추출 (엉덩이, 무릎, 어깨)
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

        # 허리 각도 계산
        angle = calculate_angle(shoulder, hip, knee)

        # 화면에 각도 표시
        cv2.putText(image, str(angle), tuple(np.multiply(hip, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 싯업 자세 감지
        if angle > MAX_ANGLE:
            cv2.putText(image, 'Sit-up: Up Position', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        elif angle < MIN_ANGLE:
            cv2.putText(image, 'Sit-up: Down Position', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    except:
        pass

    # 포즈 랜드마크를 이미지 위에 그리기
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 결과 화면에 표시
    cv2.imshow('Sit-up Pose Detection', image)

    # 'q'를 누르면 종료
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
