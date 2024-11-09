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
# from pose_eye import analyze_eye_foot  # 'pose_eye.py'에서 눈 감김 및 한 발 분석 함수 가져오기
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

    try:
        running = True
        while running:
            try:
                client_socket, client_address = server_socket.accept()
                print(f'클라이언트 연결됨: {client_address}')

                while running:
                    try:
                        # 1. 모델 선택 신호를 먼저 수신 (스쿼트: 1, 제자리 뛰기: 2, 눈 감김/발 판정: 3, 크런치: 4, 사이드스텝: 5, 킥 감지: 6)
                        model_data = recv_all(client_socket, 1)
                        if model_data is None:
                            print("모델 선택 신호 수신 실패")
                            break

                        model_choice = struct.unpack('<B', model_data)[0]
                        print(f"선택된 모델: {model_choice}")

                        # 2. 이미지 데이터 수신 및 디코딩
                        frame = receive_image_data(client_socket)
                        if frame is None:
                            print("이미지 데이터 수신 실패")
                            continue  # 이미지가 없으면 다음 루프로 이동

                        # 수신된 이미지를 화면에 표시
                        cv2.imshow("Received Frame", frame)

                        # 3. 모델에 따른 분석 수행
                        if model_choice == 1:
                            count = analyze_squat(frame)  # 스쿼트 분석
                            print(f"스쿼트 횟수: {count}")
                        elif model_choice == 2:
                            count = analyze_jump(frame)  # 제자리 뛰기 분석
                            print(f"제자리 뛰기 횟수: {count}")
                        
                        # elif model_choice == 3:
                        #     status = analyze_eye_foot(frame)  # 눈 감김 및 한 발 판정
                        #     print(f"챌린지 상태: {status}")
                        #     client_socket.sendall(status.encode('utf-8'))
                        #     continue
                        elif model_choice == 4:
                            count = analyze_crunch(frame)  # 크런치 분석
                            print(f"크런치 횟수: {count}")
                        # elif model_choice == 5:
                        #     count = analyze_side_step(frame)  # 사이드스텝 분석
                        #     print(f"사이드스텝 횟수: {count}")
                        elif model_choice == 6:
                            count = analyze_kick(frame)  # 킥 감지 분석
                            print(f"킥 횟수: {count}")
                        else:
                            print("알 수 없는 모델 선택")
                            continue

                        # count가 None이면 0으로 설정
                        count = count if count is not None else 0

                        # 결과 값을 클라이언트로 전송
                        client_socket.sendall(struct.pack('<I', int(count)))

                        # 'q' 키를 누르면 서버 종료
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            running = False
                            break

                    except socket.timeout:
                        pass  # 타임아웃 예외 발생 시 패스

            except socket.timeout:
                continue  # 타임아웃 발생 시 다시 accept 대기

    except KeyboardInterrupt:
        print("\n서버 종료 중...")

    finally:
        if client_socket:
            client_socket.close()  # 클라이언트 소켓 닫기
        server_socket.close()  # 서버 소켓 닫기
        cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기
        print("서버 소켓이 닫혔습니다.")

if __name__ == "__main__":
    start_server()
