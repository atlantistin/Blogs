"""------------------------keras-hdf5------------------------"""
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import cv2

# data
img_path = "./data/0.png"
img = cv2.imread(img_path, 0)

# model
keras_model_path = "./data/unet_membrane.hdf5"
unet = load_model(keras_model_path)

# prepare_and_predict
img = img / 255
img = cv2.resize(img, (256, 256))
X = np.reshape(img, (1, 256, 256, 1))
y = unet.predict(X)
out = np.reshape(y, (256, 256)) * 255
out = np.array(out, dtype="u1")
print("img.shape: %s, out.shape: %s" % (img.shape, out.shape))

# show
plt.subplot(1, 2, 1); plt.imshow(img, cmap="gray")
plt.subplot(1, 2, 2); plt.imshow(out, cmap="gray")
plt.suptitle("keras-hdf5"); plt.show()


"""------------------------tensorflow-pb------------------------"""
import tensorflow as tf

# model
tf_model_path = "./data/unet_membrane.pb"

# graph_and_predict
with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()

    # open the *.pb model
    with open(tf_model_path, "rb") as fr:
        output_graph_def.ParseFromString(fr.read())
        tf.import_graph_def(output_graph_def, name="")

    # run the forward in the session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # init
        inp_tensor = sess.graph.get_tensor_by_name("input_1:0")
        out_tensor = sess.graph.get_tensor_by_name("conv2d_24/Sigmoid:0")
        npy_tensor = sess.run(out_tensor, feed_dict={inp_tensor: X})  # X in line 18.
        # postprocessing
        npy = np.reshape(npy_tensor, (256, 256)) * 255
        npy = np.array(npy, dtype="u1")

# show
plt.subplot(1, 3, 1); plt.imshow(img, cmap="gray"), plt.title("IMG")
plt.subplot(1, 3, 2); plt.imshow(out, cmap="gray"), plt.title("keras")
plt.subplot(1, 3, 3); plt.imshow(npy, cmap="gray"), plt.title("TF")
plt.show()
