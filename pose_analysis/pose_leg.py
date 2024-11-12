import cv2
from mediapipe.python.solutions import pose as mp_pose
import numpy as np

# MediaPipe Pose 초기화 (복잡도 증가 및 static_image_mode 사용)
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=2,  # 더 높은 복잡도로 정확도 향상
    static_image_mode=True  # 이미지 모드로 설정
)

# 발 위치 차이 값을 저장할 리스트
foot_diff_history = []
smoothed_foot_diff = 0  # 보간된 발 위치 차이 값

# 연속된 프레임에서의 안정적인 판정을 위한 변수
stable_count = 0
stable_threshold = 5  # 안정적인 판정을 위한 임계값 (연속된 프레임 수)
last_count = 2  # 마지막으로 확인된 상태 (0: 두 발, 1: 한 발, 2: 인식 실패)

# 추가된 변수: 한 번 한 발로 판정되면 일정 시간 동안 유지하기 위한 변수
hold_one_foot_count = 0
hold_threshold = 10  # 한 번 한 발로 판정되면 최소 10프레임 동안 유지

# 최소 지속 시간 (초) 설정 - 한 번 '1'로 판정되면 최소한 이 시간 동안은 '1' 상태를 유지
min_hold_time_seconds = 5  
fps = 30  # 예상되는 프레임 속도 (예시로 초당 30프레임으로 설정)
min_hold_frames = min_hold_time_seconds * fps

# 위치 보간을 위한 Smoother 클래스 (알파 값으로 보간 정도 조절)
class Smoother:
    def __init__(self, alpha=0.05):  # 알파 값이 작을수록 더 부드럽게 보간됨
        self.alpha = alpha
        self.previous_value = None

    def smooth(self, current_value):
        if self.previous_value is None:
            self.previous_value = current_value
        else:
            self.previous_value = self.alpha * current_value + (1 - self.alpha) * self.previous_value
        return self.previous_value

# 자세 안정화를 위한 Stabilization 클래스 (window_size만큼의 상태를 확인)
class Stabilization:
    def __init__(self, threshold=3, window_size=5):
        self.threshold = threshold  # 연속된 프레임에서 같은 상태를 확인하기 위한 임계값
        self.history = np.zeros(window_size)  # 최근 'window_size'개의 상태를 저장
        self.index = 0

    def update(self, state):
        self.history[self.index] = state
        self.index = (self.index + 1) % len(self.history)

    def is_stable(self):
        # 최근 상태 중 일정 비율 이상이 같은 상태일 때만 True 반환
        return np.sum(self.history) >= self.threshold

# Smoother와 Stabilization 객체 초기화
smoother = Smoother(alpha=0.05)   # 더 부드럽게 보간하기 위해 알파 값 낮춤
stability_checker = Stabilization(window_size=5)

# 자세 분석 함수 (위치 보간 및 안정화 포함)
def analyze_foot(frame):
    global foot_diff_history, stable_count, last_count, smoothed_foot_diff, hold_one_foot_count

    try:
        count = last_count

        if frame is None:
            print("입력 이미지가 None입니다.")
            return count

        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            frame = cv2.resize(frame, (640, int(height * scale)))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        debug_frame = frame.copy()
        if results.pose_landmarks:
            # 랜드마크 그리기 (디버깅용)
            for landmark in results.pose_landmarks.landmark:
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(debug_frame, (cx, cy), 5, (0, 255, 0), -1)

            # 왼발과 오른발의 Y 좌표 추출
            left_foot_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y
            right_foot_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y

            # 발 가시성 확인
            left_foot_visibility = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].visibility
            right_foot_visibility = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].visibility

            # 가시성이 낮으면 인식 실패로 처리
            if left_foot_visibility < 0.6 or right_foot_visibility < 0.6:
                print("발 랜드마크가 명확하지 않습니다.")
                return last_count

            # 발 위치 차이 계산 및 임계값 조정 (더 정밀하게)
            foot_diff = abs(left_foot_y - right_foot_y)
            print(f"발 위치 차이: {foot_diff}")

            # 위치 보간(Smoothing) 적용하여 부드러운 값 계산
            smoothed_foot_diff = smoother.smooth(foot_diff)

            # 추가 조건: 엉덩이와 무릎 각도 계산 (예시)
            left_knee_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y
            right_knee_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y
            
            # 무릎과 엉덩이 사이의 각도 비교 강화
            knee_diff = abs(left_knee_y - right_knee_y)

            # 한 발로 서 있는지 여부 판단 (보간된 값 사용)
            if smoothed_foot_diff > 0.02 and knee_diff > 0.02:  
                stability_checker.update(1)  # 한 발로 서 있는 상태 업데이트
                
                if stability_checker.is_stable():  
                    count = 1   # 연속된 프레임에서 안정적이면 한 발로 서 있다고 확정
                    
                    hold_one_foot_count += min_hold_frames   # 최소 지속 시간 동안 유지
                    
                    cv2.putText(debug_frame, "One Foot", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            else:
                stability_checker.update(0)   # 두 발로 서 있는 상태 업데이트
                
                if stability_checker.is_stable():  
                    count = 0   # 연속된 프레임에서 안정적이면 두 발로 서 있다고 확정
                    
                    cv2.putText(debug_frame, "Two Feet", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        else:
            print("포즈 랜드마크를 찾을 수 없습니다.")
            count = last_count
        
        last_count = count
        
        # 만약 한 번 한 발로 판정되었다면 최소 지속 시간 동안 유지하도록 함
        if hold_one_foot_count > min_hold_frames:
            hold_one_foot_count -= min_hold_frames // fps   # 지속 시간 감소 처리 중단 시점까지 유지함.
        
        if hold_one_foot_count > 0:
            count = 1
        
        # 결과 시각화 출력
        cv2.imshow("Debug View", debug_frame)   # 창 이름과 이미지를 전달해야 함.
        cv2.waitKey(1)

        return count

    except Exception as e:
        print(f"분석 중 오류 발생: {e}")
        return last_count  