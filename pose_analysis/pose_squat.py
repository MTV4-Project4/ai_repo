import cv2
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
import numpy as np

pose = mp_pose.Pose()

cap = cv2.VideoCapture('C:/MTV4/mediapipe/squat.mp4')
#cap = cv2.VideoCapture(0)
width = 1280
height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

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

# 올바른 자세 판별 함수
def is_correct_squat(left_angle, right_angle):
    # 두 다리의 각도 확인 (down 상태는 100도 이하, up 상태는 110도 이상)
    if left_angle < 100 or right_angle < 100:  # down 상태
        return True  # 올바른 자세
    elif left_angle > 110 and right_angle > 110:  # up 상태
        return True  # 올바른 자세
    return False  # 잘못된 자세

# 스쿼트 자세 여부 확인 함수 (기본적인 자세 취함 여부, 좌표를 픽셀 좌표로 변환 후 확인)
def is_squat_position(left_hip, left_knee, right_hip, right_knee, img_height):
    # y 좌표 기준으로 엉덩이가 무릎보다 아래에 있어야 스쿼트 자세로 판단
    left_hip_y = left_hip[1] * img_height
    left_knee_y = left_knee[1] * img_height
    right_hip_y = right_hip[1] * img_height
    right_knee_y = right_knee[1] * img_height
    
    if left_hip_y > left_knee_y and right_hip_y > right_knee_y:
        return True
    return False

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
        
        # 이미지 크기 기반으로 좌표 계산
        img_height, img_width, _ = img.shape

        # 좌표 변환: Mediapipe 좌표를 픽셀 단위로 변환
        left_hip = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * img_width, 
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * img_height
        ]
        left_knee = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * img_width,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * img_height
        ]
        left_ankle = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x * img_width,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * img_height
        ]
        left_angle = calculate_angle(left_hip, left_knee, left_ankle)

        right_hip = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * img_width, 
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * img_height
        ]
        right_knee = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * img_width,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * img_height
        ]
        right_ankle = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x * img_width,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * img_height
        ]
        right_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
        # 스쿼트 자세인지 확인
        squat_position = is_squat_position(left_hip, left_knee, right_hip, right_knee, img_height)
        
        # 스쿼트 자세일 때만 올바른 자세 판별
        if squat_position:
            correct_squat = is_correct_squat(left_angle, right_angle)
            if correct_squat:
                print("True")  # 올바른 자세
            else:
                print("False")  # 잘못된 자세
        else:
            print("Not in Squat Position")  # 스쿼트 자세 아님

        # 화면에 올바른 자세 여부 표시
        if squat_position:
            cv2.putText(img, f"Posture: {'Correct' if correct_squat else 'Incorrect'}", 
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (0, 255, 0) if correct_squat else (0, 0, 255), 3)
        else:
            cv2.putText(img, "Not in Squat Position", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow('Squat Analysis', img)
    
    if cv2.waitKey(1) == 27:  # ESC 키로 종료
        break

cap.release()
cv2.destroyAllWindows()
