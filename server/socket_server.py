import socket
import struct
import cv2
import numpy as np

def recv_all(sock, count):
    """ 지정한 바이트 수만큼 데이터를 수신 """
    buffer = b''
    while len(buffer) < count:
        packet = sock.recv(count - len(buffer))
        if not packet:
            return None  # 연결이 끊어졌거나 문제가 발생함
        buffer += packet
    return buffer

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.settimeout(1)  # 타임아웃 설정 (1초)

    server_socket.bind(('0.0.0.0', 17232))  # 모든 인터페이스에서 접속 허용
    server_socket.listen(1)

    print('서버가 시작되었습니다. 클라이언트 연결 대기 중...')

    client_socket = None

    try:
        running = True  # 루프 제어 변수
        while running:
            try:
                # 클라이언트 연결 대기 (타임아웃 발생 시 계속 대기)
                client_socket, client_address = server_socket.accept()
                print(f'클라이언트 연결됨: {client_address}')

                while running:
                    try:
                        # 4바이트 크기의 이미지 크기 정보 먼저 수신
                        image_size_data = recv_all(client_socket, 4)
                        if image_size_data is None:
                            print("이미지 크기 정보 수신 실패")
                            break

                        # 이미지 크기 (빅 엔디안으로 변환된 정수형 해석)
                        image_size = struct.unpack("!I", image_size_data)[0]  # 빅 엔디안
                        print(f"수신한 이미지 크기: {image_size} bytes")

                        # 이미지 데이터 수신
                        image_data = recv_all(client_socket, image_size)
                        if image_data is None:
                            print("패킷 수신 중단")
                            break

                        # 수신한 이미지 데이터를 OpenCV로 디코딩
                        np_image = np.frombuffer(image_data, dtype=np.uint8)
                        frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

                        # 수신한 프레임 표시
                        if frame is not None:
                            cv2.imshow("Received Video", frame)

                        if cv2.waitKey(1) == ord('q'):
                            running = False  # 'q' 키를 누르면 서버 종료
                            break

                    except socket.timeout:
                        pass  # 타임아웃 예외 발생 시 패스

            except socket.timeout:
                continue  # 타임아웃 발생 시 다시 accept 대기

    except KeyboardInterrupt:
        print("\nCtrl + C 감지됨. 서버를 종료합니다.")
        running = False  # 루프 종료

    finally:
        # 소켓 종료
        if client_socket:
            client_socket.close()  # 클라이언트 소켓 닫기
        server_socket.close()  # 서버 소켓 닫기
        cv2.destroyAllWindows()
        print("서버 소켓이 닫혔습니다.")

if __name__ == "__main__":
    start_server()
