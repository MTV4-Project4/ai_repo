import cv2
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
import numpy as np

pose = mp_pose.Pose()

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('C:/MTV4/pushup.mp4')

width = 1280
height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

pushup_state = False  # down 상태 여부
count = 0  # 푸쉬업 카운트

# 3 점을 가지고 각도를 구하는 함수
def calculate_angle(a, b, c):  # b가 가운데
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosangle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))  # 내적 / 길이*길이
    arccos = np.arccos(cosangle)
    degree = np.degrees(arccos)  # 각도로 바꿈
    return degree


while cap.isOpened():
    ret, img = cap.read()
    img = cv2.flip(img, 1)

    if not ret:
        break

    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
        )

        # 이미지의 크기를 고려하여 좌표를 픽셀 좌표로 변환
        image_width, image_height = img.shape[1], img.shape[0]

        left_shoulder = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height
        ]
        left_elbow = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * image_width,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * image_height
        ]
        left_wrist = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image_height
        ]
        left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

        right_shoulder = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height
        ]
        right_elbow = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * image_width,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * image_height
        ]
        right_wrist = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height
        ]
        right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Debugging output for angles
        print(f"Left Angle: {left_angle}, Right Angle: {right_angle}")

        # 푸쉬업 상태 감지 및 카운팅
        if left_angle <= 90 and right_angle <= 90:
            cv2.putText(img, "down", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)
            pushup_state = True  # down 상태
        if left_angle >= 160 and right_angle >= 160:
            cv2.putText(img, "up", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)
            if pushup_state:  # 이전 상태가 down이었다면
                count += 1  # 푸쉬업 카운트 증가
                pushup_state = False  # up 상태로 변경

        # 카운트 표시
        cv2.putText(img, f"Count: {count}", (50, 200), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 3)

    # 이미지 출력
    cv2.imshow('Push-up Counter', img)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
