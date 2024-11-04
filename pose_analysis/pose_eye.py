import cv2
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import face_mesh as mp_face_mesh
import numpy as np

# MediaPipe 초기화
pose = mp_pose.Pose()
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5)

# 눈 감김 여부를 판단하는 함수
def are_eyes_closed(landmarks):
    left_eye_indices = [33, 160, 158, 133, 153, 144]  # 왼쪽 눈 외곽
    right_eye_indices = [362, 385, 387, 263, 373, 380]  # 오른쪽 눈 외곽

    left_eye_height = np.linalg.norm(
        np.array([landmarks[159].x, landmarks[159].y]) - np.array([landmarks[145].x, landmarks[145].y])
    )
    right_eye_height = np.linalg.norm(
        np.array([landmarks[386].x, landmarks[386].y]) - np.array([landmarks[374].x, landmarks[374].y])
    )

    eye_aspect_ratio_threshold = 0.03  # 눈이 감겼다고 판단하는 임계값

    return left_eye_height < eye_aspect_ratio_threshold and right_eye_height < eye_aspect_ratio_threshold

# 자세 분석 함수
def analyze_pose(frame):
    status = "bad"  # 초기 상태

    img = cv2.flip(frame, 1)
    
    # 얼굴 인식
    face_results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    eyes_closed = False
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            if are_eyes_closed(face_landmarks.landmark):
                eyes_closed = True
                break

    # 자세 인식
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        left_foot_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y
        right_foot_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y
        left_hip_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
        right_hip_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y

        # 두 발 중 하나만 땅에 닿아 있는지 판단 (약간의 오차 허용)
        if abs(left_foot_y - right_foot_y) > 0.05:
            if eyes_closed:
                status = "good"
            else:
                status = "눈x"
        else:
            if eyes_closed:
                status = "발x"
            else:
                status = "bad"

    return status


