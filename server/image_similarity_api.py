from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from io import BytesIO
import cv2
import mediapipe as mp
from scipy.spatial import distance
import uvicorn

app = FastAPI()

#이미지 URL을 받는 데이터 모델 정의
class ImageURLs(BaseModel):
    url1:str
    url2:str
    
#URL에서 이미지를 다운로드하는 함수
def fetch_image_from_url(url: str):
    response = requests.get(url)
    if response.status_code == 200:
        return cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    else:
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다")

#포즈 랜드마크 추출 함수
def extract_pose_landmarks(image):
    mp_pose = mp.solutions.pose
    pose = mp_pose.pose(static_image_mode=True)
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pose.close()
    if results.pose_landmarks:
        return [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
    else:
        return None
    
# 두 랜드마크 간의 유사도 계산 함수
def calculate_similarity(landmarks1, landmarks2):
    if len(landmarks1) != len(landmarks2):
        return 0  # 랜드마크 길이가 다르면 비교 불가능
    distances = [distance.euclidean(l1, l2) for l1, l2 in zip(landmarks1, landmarks2)]
    similarity_score = 100 - (np.mean(distances) * 100)  # 대략적인 유사도 점수 계산
    return max(0, similarity_score)  # 0보다 작은 값이 나오지 않도록 처리


@app.post("/compare_images")
async def compare_images(urls: ImageURLs):
    # 이미지 다운로드
    image1 = fetch_image_from_url(urls.url1)
    image2 = fetch_image_from_url(urls.url2)

    # MediaPipe를 사용해 랜드마크 추출
    landmarks1 = extract_pose_landmarks(image1)
    landmarks2 = extract_pose_landmarks(image2)

    if not landmarks1 or not landmarks2:
        return {"message": "한 이미지 또는 두 이미지에서 포즈를 감지하지 못했습니다."}

    # 유사도 계산
    similarity_score = calculate_similarity(landmarks1, landmarks2)
    result = "유사함" if similarity_score >= 80 else "유사하지 않음"

    return {
        "similarity_score": similarity_score,
        "result": result
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7900)