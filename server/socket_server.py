import socket
import struct
import cv2
import numpy as np


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
    try:
        # 4바이트 크기의 이미지 크기 정보 수신
        size_data = recv_all(client_socket, 4)
        if size_data is None or len(size_data) < 4:
            raise ValueError("이미지 크기 정보 수신 실패")

        # 리틀 엔디안 그대로 해석
        image_size = struct.unpack('<I', size_data)[0]
        print(f"수신한 이미지 크기: {image_size} bytes")

        # 전체 이미지 데이터를 수신
        image_data = recv_all(client_socket, image_size)
        if image_data is None or len(image_data) != image_size:
            raise ValueError("전체 이미지 데이터를 수신하지 못했습니다.")

        return image_data

    except Exception as e:
        print(f"이미지 수신 중 오류 발생: {e}")
        return None

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.settimeout(1)

    server_socket.bind(('0.0.0.0', 17232))
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
                        image_data = receive_image_data(client_socket)
                        if image_data is None:
                            print("이미지 데이터 수신 실패")
                            break

                        np_image = np.frombuffer(image_data, dtype=np.uint8)
                        frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

                        # 수신한 프레임을 화면에 표시
                        if frame is not None:
                            cv2.imshow("Received Image", frame)

                        if cv2.waitKey(1) == ord('q'):
                            running = False
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
