from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import glob as gb
import shutil
import cv2
import sys
import os

# 汉字集图像尺寸
height, width = 64, 64
ziti_size = height // 4 * 3
hanzi_size = (height, width)
hanzi_shape = (height, width, 3)

# 准备生成目录
data_directory = "data"
if os.path.exists(data_directory):
    shutil.rmtree(data_directory)
    os.makedirs(data_directory)
    
# 背景准备
back_directory = "back"
def gener_back():
    back_imgpaths = gb.glob(os.path.join(back_directory, "*"))
    total_bakimgs = len(back_imgpaths)
    while True:
        yield Image.open(back_imgpaths[np.random.randint(total_bakimgs)]).resize(hanzi_size)

# 汉字生成
ziti_directory = "ziti"
def gener_ziti(zi, n=3):
    ziti_paths = gb.glob(os.path.join(ziti_directory, "*"))
    ziti_total = len(ziti_paths)
    while True:
        if n <= 0:
            break
        else:
            n -= 1
        # 添加字
        _img = Image.fromarray(np.zeros(hanzi_shape, dtype="u1"))
        font = ImageFont.truetype(ziti_paths[np.random.randint(ziti_total)], ziti_size, encoding="utf-8")
        r, g, b = np.random.randint(150, 255), np.random.randint(150, 255), np.random.randint(150, 255)
        draw = ImageDraw.Draw(_img)
        draw.text((5, 5), zi, (r, g, b), font=font)
        # 若不使用旋转可注释掉
        _img = _img.rotate(np.random.randint(-45, 45))
        # 若不使用模糊可注释掉
        _img = _img.filter(ImageFilter.GaussianBlur(radius=0.7))
        # 若不使用错切可注释掉
        theta = np.random.randint(-15, 15) * np.pi / 180
        M_shear = np.array([[1, np.tan(theta), 0], [0, 1, 0]], dtype=np.float32)
        _img = Image.fromarray(cv2.warpAffine(np.array(_img), M_shear, hanzi_size))
        yield _img

# 数据集
data_directory = "data"
with open("GB2312.txt", 'r', encoding="utf-8") as fr:
    zi_sets = fr.read()
for i, zi in enumerate(zi_sets):
    # 目录准备
    zi_directory = os.path.join(data_directory, zi)
    if not os.path.exists(zi_directory):
        os.makedirs(zi_directory)
    # 开始生成
    for serial, (ziti, back) in enumerate(zip(gener_ziti(zi, n=5), gener_back())):  # n=5, 生成5张/字体
        img = Image.fromarray(np.array(ziti) // 5 * 3 + np.array(back) // 5 * 2)
        img_path = os.path.join(zi_directory, str(serial) + ".jpg")
        img.save(img_path, "JPEG")
    # 调试时用于控制生成汉字种类个数
    if i > 3:
        break