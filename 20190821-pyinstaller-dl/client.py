import socket


# 客户端
def cc():
    HOST, PORT = '127.0.0.1', 8888
    socketclient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socketclient.connect((HOST, PORT))
    while True:
        try:
            reques = input("请输入图像路径: ")
            if reques == "q":
                break
            socketclient.send(reques.encode())
            result = socketclient.recv(1024).decode()
            print(result)
        except:
            pass


if __name__ == "__main__":
     cc()