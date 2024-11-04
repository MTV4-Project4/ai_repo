import socket
import struct
import sys
import os

# 'pose_analysis' 디렉터리 경로를 Python 모듈 경로에 추가
sys.path.append(os.path.abspath('C:/MTV4/pose_analysis'))

from pose_squat2 import analyze_pose as analyze_squat  # 'pose_squat2.py'에서 스쿼트 분석 함수 가져오기
from pose_jump import analyze_jump  # 'pose_jump.py'에서 제자리 뛰기 분석 함수 가져오기
from pose_eye import analyze_eye_foot  # 'pose_eye.py'에서 눈 감김 및 한 발 분석 함수 가져오기
from pose_crunch import analyze_pose as analyze_crunch  # 'pose_crunch.py'에서 크런치 분석 함수 가져오기
from pose_side_step import analyze_pose as analyze_side_step  # 'pose_side_step.py'에서 사이드스텝 분석 함수 가져오기

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
                        # 1. 모델 선택 신호를 먼저 수신 (스쿼트: 1, 제자리 뛰기: 2, 눈 감김/발 판정: 3, 크런치: 4, 사이드스텝: 5)
                        model_data = recv_all(client_socket, 1)
                        if model_data is None:
                            print("모델 선택 신호 수신 실패")
                            break

                        model_choice = struct.unpack('<B', model_data)[0]
                        print(f"선택된 모델: {model_choice}")

                        # 2. 모델에 따른 분석 수행
                        if model_choice == 1:
                            count = analyze_squat()  # 스쿼트 분석
                            print(f"스쿼트 횟수: {count}")
                            # Unity로 count 값을 전송
                            client_socket.sendall(struct.pack('<I', count))
                        elif model_choice == 2:
                            count = analyze_jump()  # 제자리 뛰기 분석
                            print(f"제자리 뛰기 횟수: {count}")
                            # Unity로 count 값을 전송
                            client_socket.sendall(struct.pack('<I', count))
                        elif model_choice == 3:
                            status = analyze_eye_foot()  # 눈 감김 및 한 발 판정
                            print(f"챌린지 상태: {status}")
                            # Unity로 상태를 전송
                            client_socket.sendall(status.encode('utf-8'))
                        elif model_choice == 4:
                            count = analyze_crunch()  # 크런치 분석
                            print(f"크런치 횟수: {count}")
                            # Unity로 count 값을 전송
                            client_socket.sendall(struct.pack('<I', count))
                        elif model_choice == 5:
                            count = analyze_side_step()  # 사이드스텝 분석
                            print(f"사이드스텝 횟수: {count}")
                            # Unity로 count 값을 전송
                            client_socket.sendall(struct.pack('<I', count))
                        else:
                            print("알 수 없는 모델 선택")

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
        print("서버 소켓이 닫혔습니다.")

if __name__ == "__main__":
    start_server()
