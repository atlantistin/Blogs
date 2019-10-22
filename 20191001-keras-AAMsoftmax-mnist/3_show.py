from aam_softmax_model import build_net
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.models import Model
import pandas as pd
import numpy as np
import glob as gb
import os

# 准备
data = []
model_path = "./model.h5"
datasets_path = "./datasets/mnist_test"
# 加载
model = build_net()
model.load_weights(model_path)
vec_layer = Model(inputs=model.input, outputs=model.get_layer("feature_embedding").output)
# 计算
category_path = gb.glob(os.path.join(datasets_path, "**/"))
for category in category_path:
    y = int(category.split(os.sep)[-2])
    for idx, img_path in enumerate(gb.glob(os.path.join(category, "*"))):
        if idx >= 5:
            break
        img = image.load_img(img_path,
                             target_size=(28, 28),
                             grayscale=True,
                             )
        x = image.img_to_array(img)
        x *= 1./255
        x = np.expand_dims(x, axis=0)
        x2vec = vec_layer.predict(x)
        X_y = x2vec.flatten()
        X_y /= np.linalg.norm(X_y)  # 归一化：可不进行归一化即注释后对比观察
        X_y = X_y.tolist()
        X_y.append(y)
        data.append(X_y)
data = pd.DataFrame(data)
# 存储
# data.to_csv("vec.csv", sep='\t', header=False, index=False, float_format="%.7f")
# 显示
data[2] = data[2].astype('int')
classes = []
colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:olive', 'tab:cyan', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
markers = ['v', ',', 'o', '^', '.', '<', '>', 'p', '*', 'x']
for i in range(data[2].max() + 1):
    classes.append(data.loc[data[2] == i])
    plt.scatter(classes[i][0], classes[i][1], color=colors[i], marker=markers[i])
plt.title("Vector!")
plt.axis("square")
plt.show()
