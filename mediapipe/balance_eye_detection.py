import cv2
import mediapipe as mp
import numpy as np

# Mediapipe 설정
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 눈을 감았는지 확인하는 임계값 (0.2 이하면 눈을 감은 상태로 간주)
EYE_AR_THRESH = 0.2

# 왼쪽 눈, 오른쪽 눈 랜드마크
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

# 한 발로 서 있는지 판정하기 위한 최소 발 차이
FOOT_LIFT_THRESHOLD = 0.1

# 눈 감기 감지 함수
def eye_aspect_ratio(landmarks, image_height, image_width):
    """눈 감기 정도를 측정하는 함수"""
    left_eye = np.array([(landmarks[pt].x * image_width, landmarks[pt].y * image_height) for pt in LEFT_EYE_LANDMARKS])
    right_eye = np.array([(landmarks[pt].x * image_width, landmarks[pt].y * image_height) for pt in RIGHT_EYE_LANDMARKS])

    # 눈 높이 계산 (상하)
    left_eye_height = np.linalg.norm(left_eye[1] - left_eye[5])
    right_eye_height = np.linalg.norm(right_eye[1] - right_eye[5])

    # 눈 너비 계산 (좌우)
    left_eye_width = np.linalg.norm(left_eye[0] - left_eye[3])
    right_eye_width = np.linalg.norm(right_eye[0] - right_eye[3])

    # 눈의 종횡비 (aspect ratio) 계산
    left_eye_ar = left_eye_height / left_eye_width
    right_eye_ar = right_eye_height / right_eye_width

    return (left_eye_ar + right_eye_ar) / 2

# 얼굴을 크롭하는 함수
def crop_face(image, landmarks, image_height, image_width):
    """얼굴을 찾아서 크롭하고 확대하는 함수"""
    face_coords = [(int(landmark.x * image_width), int(landmark.y * image_height)) for landmark in landmarks]
    x_coords, y_coords = zip(*face_coords)
    x_min, x_max = max(min(x_coords) - 10, 0), min(max(x_coords) + 10, image_width)
    y_min, y_max = max(min(y_coords) - 10, 0), min(max(y_coords) + 10, image_height)
    
    cropped_face = image[y_min:y_max, x_min:x_max]
    return cropped_face

# 비디오 캡처
cap = cv2.VideoCapture(0)

# Mediapipe 모델 로드
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
        mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("비디오 프레임을 읽을 수 없습니다.")
            break

        # 이미지를 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Mediapipe 모델 적용 (FaceMesh & Pose)
        face_results = face_mesh.process(image)
        pose_results = pose.process(image)

        # 이미지를 다시 BGR로 변환
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image_height, image_width, _ = image.shape

        eye_closed = False
        foot_lifted = False

        # 눈 감김 상태 확인 및 얼굴 크롭
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                eye_ar = eye_aspect_ratio(face_landmarks.landmark, image_height, image_width)

                # 눈 감김 여부 판단
                if eye_ar < EYE_AR_THRESH:
                    eye_status = 'Eyes Closed'
                    eye_closed = True
                else:
                    eye_status = 'Eyes Open'
                    eye_closed = False

                # 얼굴 크롭
                cropped_face = crop_face(image, face_landmarks.landmark, image_height, image_width)
                
                # 크롭된 얼굴 확대
                resized_face = cv2.resize(cropped_face, (200, 200))  # 얼굴을 확대
                
                # 확대된 얼굴을 원본 이미지 좌측 상단에 삽입
                face_height, face_width, _ = resized_face.shape
                image[10:10+face_height, 10:10+face_width] = resized_face
                
                # 눈 감김 여부 표시
                cv2.putText(image, eye_status, (10, face_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 한 발로 서 있는 상태 확인
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark

            # 왼쪽 발목과 오른쪽 발목의 y 좌표 추출
            left_ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
            right_ankle_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y

            if abs(left_ankle_y - right_ankle_y) > FOOT_LIFT_THRESHOLD:
                balance_status = 'One Foot Balance'
                foot_lifted = True
            else:
                balance_status = 'Both Feet on Ground'
                foot_lifted = False

            # 전체 화면에 다리 상태 표시
            cv2.putText(image, balance_status, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # 포즈 랜드마크 그리기
            mp_drawing.draw_landmarks(
                image, 
                pose_results.pose_landmarks,  # 포즈 랜드마크를 그릴 대상
                mp_pose.POSE_CONNECTIONS  # 포즈 연결
            )

        # 눈 감김 상태와 발 상태를 모두 만족할 때 화면에 연두색 표시
        if eye_closed and foot_lifted:
            status = 'Eyes Closed & One Foot Balance'
            status_color = (0, 255, 0)  # 연두색
        else:
            status = ''
            status_color = (0, 0, 0)  # 기본 색상

        if status:
            cv2.putText(image, status, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)

        # 화면 확대
        enlarged_image = cv2.resize(image, (int(image_width * 1.5), int(image_height * 1.5)))  # 1.5배 확대

        # 확대된 화면 표시
        cv2.imshow('Balance & Eyes Detection', enlarged_image)

        # 'q'를 누르면 종료
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
