import os
import cv2
import shutil
import glob as gb
import numpy as np
from keras.datasets import mnist

# load_data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, X_test.shape)

# len(train) == 60000 ---> 40k for train and 20k for validation
shutil.rmtree("datasets")
train_directory = os.path.join("datasets", "mnist_train")
validation_directory = os.path.join("datasets", "mnist_validation")
test_directory = os.path.join("datasets", "mnist_test")
# train and validation
for idx, label in enumerate(y_train):
    if idx < 40000:
        label_path = os.path.join(train_directory, str(label))
    else:
        label_path = os.path.join(validation_directory, str(label))
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    img = X_train[idx]
    img_path = os.path.join(label_path, str(idx) + ".jpg")
    cv2.imwrite(img_path, img)
# test
for idx, label in enumerate(y_test):
    label_path = os.path.join(test_directory, str(label))
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    img = X_test[idx]
    img_path = os.path.join(label_path, str(idx) + ".jpg")
    cv2.imwrite(img_path, img)