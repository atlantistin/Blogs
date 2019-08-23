from keras.applications.inception_v3 import InceptionV3
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PIL import Image
import numpy as np
import time
import json
import os
import sys
import threading
import socketserver
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Handler
class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        # 创建连接
        address, port = self.client_address
        servlocaltime = time.asctime(time.localtime(time.time()))
        print("\n时间 {0} 成功创建新连接 {1}: {2} ...".format(servlocaltime, address, port))
        # 预测处理
        while True:
            try:  # 捕获客户端的意外退出
                imgpath = self.request.recv(1024).decode().strip()
                try:  # 捕获错误路径输入
                    input_X = np.expand_dims(np.array(Image.open(imgpath).resize((299, 299))), axis=0) / 255.0
                except Exception as e:
                    message = "IMGPathError: " + str(e)
                    self.request.send(message.encode())
                    continue
                ys = model.predict(input_X)
                top_3 = [(classes[str(i)], ys.flatten()[i]) for i in ys.flatten().argsort()[-1:-4:-1]]
                self.request.send(str(top_3).encode())
            except:
                self.request.close()
        
            
class MySystemTrayIcon(QSystemTrayIcon):
    def __init__(self, parent=None):   
        super(MySystemTrayIcon, self).__init__(parent, icon=QIcon("./data/taiji.ico"))
        self.show()
        self.popup_menu()
    
    def popup_menu(self):
        # 创建菜单
        self.menu = QMenu()
        self.action_about = QAction("关于", triggered=self.onAbout)
        self.action_exit = QAction("退出", triggered=self.onExit)
        # 绑定事件
        self.menu.addAction(self.action_about)
        self.menu.addAction(self.action_exit)
        # 菜单关联
        self.setContextMenu(self.menu)
    
    def onAbout(self):
        text = "我是谁!\n我来自哪!\n我要到哪去!"
        self.showMessage("关于", text)
    
    def onExit(self, event):
        self.setVisible(False)
        qApp.quit()
        os._exit(0)
        
        
# main
if __name__ == "__main__":
    # 模型加载
    model = InceptionV3()
    with open("./data/imagenet_class_index.json") as fr:
        classes = json.load(fr)
    model.predict(np.expand_dims(np.zeros((299, 299, 3), dtype="u1"), axis=0) / 255.0)
    # 创建服务
    IP, PORT = "127.0.0.1", 8888
    print("服务器地址%s:%s." % (IP, PORT))
    server = socketserver.ThreadingTCPServer((IP, PORT), Handler)
    threading_server = threading.Thread(target=server.serve_forever)
    threading_server.start()  
    # 托盘图标
    app = QApplication(sys.argv)
    QApplication.setQuitOnLastWindowClosed(False)
    systray_icon = MySystemTrayIcon()    
    systray_icon.show()
    sys.exit(app.exec_())
    
    