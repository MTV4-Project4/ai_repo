import socket
import struct
import cv2
import numpy as np
import sys
import os

# 'pose_analysis' 디렉터리 경로를 Python 모듈 경로에 추가
sys.path.append(os.path.abspath('C:/MTV4/pose_analysis'))

from pose_squat2 import analyze_pose as analyze_squat  # 'pose_squat2.py'에서 스쿼트 분석 함수 가져오기
from pose_jump import analyze_jump  # 'pose_jump.py'에서 제자리 뛰기 분석 함수 가져오기
from pose_leg import analyze_foot  # 'pose_leg.py'에서 한 발 분석 함수 가져오기
from pose_crunch import analyze_pose as analyze_crunch  # 'pose_crunch.py'에서 크런치 분석 함수 가져오기
# from pose_side_step import analyze_pose as analyze_side_step  # 'pose_side_step.py'에서 사이드스텝 분석 함수 가져오기
from pose_kick import analyze_kick  # 'pose_kick.py'에서 킥 감지 분석 함수 가져오기

def recv_all(sock, count):
    """ 지정한 바이트 수만큼 데이터를 수신 """
    buffer = b''
    while len(buffer) < count:
        try:
            packet = sock.recv(count - len(buffer))
            if not packet:
                return None  # 연결이 끊어졌거나 문제가 발생함
            buffer += packet
        except socket.timeout:
            print("소켓 타임아웃 발생")
            return None
        except Exception as e:
            print(f"데이터 수신 중 오류 발생: {e}")
            return None
    return buffer

def receive_image_data(client_socket):
    """이미지 데이터를 수신하여 OpenCV 형식으로 반환"""
    try:
        # 4바이트 크기의 이미지 크기 정보 수신
        size_data = recv_all(client_socket, 4)
        if size_data is None or len(size_data) < 4:
            raise ValueError("이미지 크기 정보 수신 실패")

        # 이미지 크기 정보 해석
        image_size = struct.unpack('<I', size_data)[0]
        print(f"수신한 이미지 크기: {image_size} bytes")

        # 이미지 데이터 수신
        image_data = recv_all(client_socket, image_size)
        if image_data is None or len(image_data) != image_size:
            raise ValueError("전체 이미지 데이터를 수신하지 못했습니다.")

        # OpenCV 형식으로 디코딩
        frame = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)

        # 디버그: 수신된 이미지가 있는지 확인
        if frame is not None:
            print("이미지 디코딩 성공: ", frame.shape)
        else:
            print("이미지 디코딩 실패")
        return frame

    except Exception as e:
        print(f"이미지 수신 중 오류 발생: {e}")
        return None

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.settimeout(1)  # 타임아웃 설정

    server_socket.bind(('0.0.0.0', 17232))  # 모든 인터페이스에서 접속 허용
    server_socket.listen(1)

    print('서버가 시작되었습니다. 클라이언트 연결 대기 중...')

    client_socket = None
    previous_model = None  # 이전 모델 선택을 저장할 변수

    try:
        running = True
        while running:
            try:
                client_socket, client_address = server_socket.accept()
                print(f'클라이언트 연결됨: {client_address}')

                while running:
                    try:
                        # 모델 선택 신호를 먼저 수신 (스쿼트: 1, 제자리 뛰기: 2, 눈 감김/발 판정: 3, 크런치: 4, 사이드스텝: 5, 킥 감지: 6)
                        model_data = recv_all(client_socket, 1)
                        if model_data is None:
                            print("모델 선택 신호 수신 실패")
                            break

                        model_choice = struct.unpack('<B', model_data)[0]
                        print(f"선택된 모델: {model_choice}")

                        # 새로운 모델이 선택되었을 때 초기화
                        if previous_model != model_choice:
                            if model_choice == 6:  # 킥 감지 모델 초기화 필요 시 수행
                                from pose_kick import reset_counter
                                reset_counter()
                            previous_model = model_choice

                        # 이미지 데이터 수신 및 디코딩
                        frame = receive_image_data(client_socket)
                        if frame is None:
                            print("이미지 데이터 수신 실패")
                            continue  # 이미지가 없으면 다음 루프로 이동

                        # 수신된 이미지를 화면에 표시 (디버깅용)
                        cv2.imshow("Received Frame", frame)

                        # 모델에 따른 분석 수행 (스쿼트, 제자리 뛰기 등)
                        if model_choice == 1:
                            count = analyze_squat(frame)  
                            print(f"스쿼트 횟수: {count}")
                            client_socket.sendall(struct.pack('<I', int(count)))
                            continue

                        elif model_choice == 2:
                            count = analyze_jump(frame)  
                            print(f"제자리 뛰기 횟수: {count}")
                            client_socket.sendall(struct.pack('<I', int(count)))
                            continue

                        elif model_choice == 3:
                             count = analyze_foot(frame)  
                             print(f"챌린지 상태: {count}")
                             client_socket.sendall(struct.pack('<I', int(count)))   # 정수형 상태 전송
                             continue

                        elif model_choice == 4:
                            count = analyze_crunch(frame)  
                            print(f"크런치 횟수: {count}")
                            client_socket.sendall(struct.pack('<I', int(count)))
                            continue

                        elif model_choice == 6:
                            count = analyze_kick(frame)  
                            print(f"킥 횟수: {count}")
                            client_socket.sendall(struct.pack('<I', int(count)))
                            continue

                    except ConnectionResetError as cre:
                        print(f"연결이 강제로 종료되었습니다: {cre}")
                        break

                    except socket.timeout:
                        pass  

            except socket.timeout:
                continue  

    except KeyboardInterrupt:
        print("\n서버 종료 중...")

    finally:
        if client_socket:
            client_socket.close() 
        server_socket.close()  
        cv2.destroyAllWindows()  
        print("서버 소켓이 닫혔습니다.")

if __name__ == "__main__":
    start_server()
