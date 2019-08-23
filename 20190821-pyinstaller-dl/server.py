from keras.applications.inception_v3 import InceptionV3
from PIL import Image
import numpy as np
import time
import json
import os
import wx
import wx.adv
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
        
            
# TaskBarIcon
class MyTaskBarIcon(wx.adv.TaskBarIcon):
    def __init__(self):
        self.ICON = "./data/taiji.ico"  # 图标地址
        self.ID_ABOUT = wx.NewId()  # 菜单选项"关于"的ID
        self.ID_EXIT = wx.NewId()  # 菜单选项"退出"的ID
        self.TITLE = "Inception_V3_Classify_Server"  # 鼠标移动到图标上显示的文字
        wx.adv.TaskBarIcon.__init__(self)
        self.SetIcon(wx.Icon(self.ICON), self.TITLE)  # 设置图标和标题
        self.Bind(wx.EVT_MENU, self.onAbout, id=self.ID_ABOUT)  # 绑定"关于"选项的点击事件
        self.Bind(wx.EVT_MENU, self.onExit, id=self.ID_EXIT)  # 绑定"退出"选项的点击事件
        
    # 选项"关于"的事件处理器
    def onAbout(self, event):
        wx.MessageBox("我是谁!\n我来自哪!\n我要到哪去!", "关于")
        
    # 选项"退出"的事件处理器
    def onExit(self, event):
        self.Destroy()  # 自我摧毁
        wx.Exit()
        os._exit(0)
        
    # 创建菜单选项
    def CreatePopupMenu(self):
        menu = wx.Menu()
        for mentAttr in self.getMenuAttrs():
            menu.Append(mentAttr[1], mentAttr[0])
        return menu
        
    # 获取菜单属性
    def getMenuAttrs(self):
        return [('关于', self.ID_ABOUT), ('退出', self.ID_EXIT)]
     
     
# Frame
class MyFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self)
        MyTaskBarIcon()  # 显示系统托盘图标
        threading_server = threading.Thread(target=server.serve_forever)
        threading_server.start()            


# App
class MyApp(wx.App):
    def OnInit(self):
        MyFrame()
        return True
    
     
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
    app = MyApp()
    app.MainLoop()
    
    