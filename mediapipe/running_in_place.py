 # https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md
import mediapipe as mp
import cv2
import numpy as np

# def is_full_body_visible(landmarks):
#     # 필수 랜드마크 인덱스 (양쪽 어깨, 엉덩이, 무릎, 발목)
#     required_landmarks = [
#         mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
#         mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
#         mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
#     ]
    
#     for landmark in required_landmarks:
#         if landmarks.landmark[landmark].visibility < 0.3:  # 가시성이 낮으면 감지되지 않은 것으로 간주
#             return False
#     return True
     
def calculate_angle(a,b,c): # b가 가운데
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a-b
    bc = c-b
    
    cosangle = np.dot(ba,bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)) # 내적 / 길이*길이
    arccos = np.arccos(cosangle)
    degree = np.degrees(arccos) # 각도로 바꿈
    
    return degree # 몇 도인지 각도를 리턴

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose()


left_state, right_state = False, False
cnt = 0
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)


while cap.isOpened():
    ret, img = cap.read()
    
    if not ret:
        break
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    # 결과 33개 좌표 값
    result = pose.process(img_rgb) # rgb로 넣어줘야 한다 

    # print(result.pose_landmarks)


    # 인식이 됐으면
    if result.pose_landmarks:
        # if is_full_body_visible(result.pose_landmarks):
        mp_drawing.draw_landmarks(
            img,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing_styles.get_default_pose_landmarks_style()
        )
        # mp_drawing.plot_landmarks(result.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

        ######################왼쪽################################
        # 23
        left_hip = [
            result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x, 
            result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y,
            result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].z
        ]
            
        # 25
        left_knee = [
            result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x,
            result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y,
            result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].z
        ]
        
        # 27
        left_ankle = [
            result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x,
            result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y,
            result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].z
        ]
        
        left_angle = calculate_angle(left_hip, left_knee, left_ankle)        
        
        ######################오른쪽##################################
        # 24
        right_hip = [
            result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x, 
            result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y, 
            result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].z
        ]
            
        # 26
        right_knee = [
            result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x,
            result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y,
            result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].z
        ]
        
        # 28
        right_ankle = [
            result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
            result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y,
            result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].z
        ]
        
        right_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
        if left_angle < 50:
            print(f'왼쪽: {left_angle}')
            left_state = True
        elif left_angle > 120:
            if left_state == True:
                cnt += 1
                left_state = False
            
        if right_angle < 50:
            print(f'오른쪽: {right_angle}')
            right_state = True
        elif right_angle > 120:
            if right_state == True:
                cnt += 1
                right_state = False 
        
        cv2.putText(img, str(cnt), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 5)
        # else:
        #     cv2.putText(img, "Show your whole body", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.imshow('face_mesh', img)
        
    if cv2.waitKey(1)==27:
        break
cv2.destroyAllWindows()
        
        
   