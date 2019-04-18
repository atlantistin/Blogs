import numpy as np
import math
import cv2

# read and show
img = cv2.imread("panoramagram.png")
cv2.imshow("panoramagram", img)

# get the center
x0 = img.shape[0] // 2
y0 = img.shape[1] // 2

# define unwrapped
unwrapped_height = radius = img.shape[0] // 2
unwrapped_width = int(2 * math.pi * radius)
unwrapped_img = np.zeros((unwrapped_height, unwrapped_width, 3), dtype="u1")

# traversing
except_count = 0
for j in range(unwrapped_width):
    theta = 2 * math.pi * (j / unwrapped_width)  # start position such as "+ math.pi"
    for i in range(unwrapped_height):
        unwrapped_radius = radius - i  # don't forget
        x = unwrapped_radius * math.cos(theta) + x0  # "sin" is clockwise but "cos" is anticlockwise
        y = unwrapped_radius * math.sin(theta) + y0
        x, y = int(x), int(y)
        try:
            unwrapped_img[i, j, :] = img[x, y, :]
        except Exception as e:
            except_count = except_count + 1
print(except_count)
cv2.imwrite("unwrapped.jpg", unwrapped_img)
cv2.imshow("Unwrapped", unwrapped_img)
# cv2.waitKey(0)

# plt
import matplotlib.pyplot as plt

for j in range(unwrapped_width):
    theta = 2 * math.pi * (j / unwrapped_width) - 1 / 2 * math.pi  # + math.pi
    for i in range(unwrapped_height):
        unwrapped_radius = radius - i
        x = unwrapped_radius * math.sin(theta) + x0  # "sin" is clockwise
        y = unwrapped_radius * math.cos(theta) + y0
        x, y = int(x), int(y)
        try:
            unwrapped_img[i, j, :] = img[x, y, :]
        except:
            continue

plt.subplot(2, 1, 1); plt.imshow(img[:, :, ::-1])
plt.subplot(2, 1, 2); plt.imshow(unwrapped_img[:, :, ::-1])
plt.show()