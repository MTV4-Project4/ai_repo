from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

app = FastAPI()

# 이미지 URL을 받는 데이터 모델 정의
class ImageURLs(BaseModel):
    url1: str
    url2: str

# URL에서 이미지를 다운로드하는 함수
def fetch_image_from_url(url: str):
    response = requests.get(url)
    if response.status_code == 200:
        return cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    else:
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다")

# 포즈 랜드마크 추출 및 이미지에 랜드마크 그리기 함수
def extract_and_draw_pose_landmarks(image):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
            return landmarks, image
        else:
            return None, image

# 각도 계산 함수
def calculate_angles(landmarks):
    def angle_between_points(a, b, c):
        ab = np.array(b) - np.array(a)
        cb = np.array(b) - np.array(c)
        cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    
    # 주요 관절의 각도 계산 예시 (오른쪽 어깨-팔꿈치-손목)
    right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
    right_elbow = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value]
    right_wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]

    angle_right_arm = angle_between_points(right_shoulder, right_elbow, right_wrist)

    return [angle_right_arm]

# 각도 유사도 비교 함수
def calculate_similarity(angles1, angles2):
    if len(angles1) != len(angles2):    
        return 0
    distances = [abs(a1 - a2) for a1, a2 in zip(angles1, angles2)]
    similarity_score = 100 - (np.mean(distances))
    return max(0, similarity_score)

# 이미지 유사도 분석 함수
def perform_similarity_analysis(urls: ImageURLs):
    print("이미지 유사도 분석 중입니다...")

    image1 = fetch_image_from_url(urls.url1)
    image2 = fetch_image_from_url(urls.url2)

    landmarks1, processed_image1 = extract_and_draw_pose_landmarks(image1)
    landmarks2, processed_image2 = extract_and_draw_pose_landmarks(image2)

    if not landmarks1 or not landmarks2:
        print("한 이미지 또는 두 이미지에서 포즈를 감지하지 못했습니다.")
        return {"message": "한 이미지 또는 두 이미지에서 포즈를 감지하지 못했습니다."}

    angles1 = calculate_angles(landmarks1)
    angles2 = calculate_angles(landmarks2)

    similarity_score = calculate_similarity(angles1, angles2)
    result = "유사함" if similarity_score >= 80 else "유사하지 않음"

    print(f"유사도 분석 완료 - 유사도 점수: {similarity_score}, 결과: {result}")

    # 이미지 시각화 (비동기 처리)
    from threading import Thread

    def show_images():
        cv2.imshow("Processed Image 1", processed_image1)
        cv2.imshow("Processed Image 2", processed_image2)
        cv2.waitKey(5000)  # 5초 동안 표시
        cv2.destroyAllWindows()

    Thread(target=show_images).start()

    return {
        "similarity_score": similarity_score,
        "result": result
    }

@app.post("/compare_images")
async def compare_images(urls: ImageURLs):
    return perform_similarity_analysis(urls)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7900)
    #metaai3.iptime.org:7900/compare_images