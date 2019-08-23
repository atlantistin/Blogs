# 20190822-使用keras预训练模型进行图像识别并利用pyinstaller打包程序为server.exe且启动后在状态栏显示

## 1. 前言

目前深度学习相关应用的部署还达不到传统程序的方便，尤其一些老旧的桌面应用，如果要进行全部升级难免耗资巨大，因此能简单地将python完成程序能通过打包为服务，然后客户端仅需要添加基于socket的请求部分就可以非常方便地进行算法升级，不失为一种折中的办法。

## 2. 准备

### 2.1 预训练模型的使用

此处参考博文[《keras 使用预训练模型 inception_v3 识别图片》](https://blog.csdn.net/nima1994/article/details/79942544) 不做太多叙述。

### 2.2 环境

```python
h5py==2.9.0
html5lib==0.9999999
Keras==2.1.5
Markdown==3.1.1
numpy==1.15.3
Pillow==4.1.1
protobuf==3.9.1
PyInstaller==3.3
pywin32==224
PyYAML==5.1.2
requests==2.22.0
scipy==1.3.1
six==1.12.0
tensorflow==1.4.0
tensorflow-tensorboard==0.4.0
wxPython==4.0.6

```



里面需重点注意的是 keras, tensorflow, numpy, pillow 这几个包，有时版本对应不上容易报错，个人此前就遇到过 pillow 带来的问题，整个程序完成后能在命令行顺利使用，但打包过后的程序却总是不能正确处理图像路径，一开始还以为是打包过程中出了纰漏，但结果发现只需要将 pillow 的版本进行更正即可。

### 2.3 wxPython与系统托盘图标

此处参考了另一篇文章《Python利用wxPython在系统托盘显示图标》，不方便贴外站链接，如果感兴趣可以搜到。

## 3. 服务端server.py

### 3. 1 代码

```python
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
    
    
```

* 为了实现高并发需求，直接使用了socketserver库，比较方便；
* 模型加载成功后即进行一次predict调用可以即可能避免出错，且完成一次加载后不会再出现首次请求耗费大量时间，如果想调用自己的一个或多个模型可安排规划外部模块，在外部模块的初始化函数中完成模型自身的加载和首次推理；
* 此处使用了本地IP和8888端口，可根据需要进行变更；
* 注意所用的图标，地址对应过去应当有*.ico文件。

## 4. 客户端

### 4.1 代码

```python
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
```

同样利用socket进行通信，通过在客户端传入正确的图像路径就可以返回推理的结果。此处直接使用了一般的字符串，但在实际应用过程中可能面临各种参数的调整，可以设计使用json字符串作为接口。

### 5. 演示

![](.\data\server.png)

可以看到当运行启动server.py后会在托盘处显示我们预定义好的图标。

## 6. 打包

* 打包工具用 pyinstaller，网上有很多教程可以搜索到；
* 对打包后的server.exe不希望再看到黑框可以通过打包参数进行控制，则启动后只有托盘图标驻留，但可能希望得到运行日志，通过使用 logging 包即可达到此需求；
* 不少人有探讨打包后的应用过大问题，本文 server.py 的打包结果大约在 200 M，有试过一些网友的方法例如逐个注释并排查某个包的占用问题，当然还有建虚拟环境避免Anaconda依赖繁复导致的污染，不过最后发现主要是 numpy占用高，还有的建议尽量用 ```from ... import ...```代替```import ...``` 来减少占用量，但似乎对 numpy 不管用，且其它科学计算库如 pandas、以及tensorflow 等貌似都有调用到，所以哪位朋友还有新的好方法一定说出来帮帮大家。^_^ 

## 7. 附注

文中代码已上传至：https://github.com/atlantistin/Blogs/tree/master/20190821-pyinstaller-dl
