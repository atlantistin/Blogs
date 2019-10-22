# 参考1: https://keras.io/zh/layers/writing-your-own-keras-layers/
# 参考2: https://github.com/4uiiurz1/keras-arcface
from keras.layers import Layer, InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import regularizers, BatchNormalization
from keras.models import Sequential
from keras.utils import plot_model
from keras.optimizers import Adam
from keras import backend as K


# Layer
class ArcLayer(Layer):
    def __init__(self, output_dim, s=30.0, m=0.5, regularizer=None, **kwargs):  # 初始化
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ArcLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.s = s
        self.m = m
        self.W = None
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):  # 定义本层的权
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.W = self.add_weight(name="kernel",
                                 shape=(input_dim, self.output_dim),
                                 initializer='glorot_uniform',
                                 regularizer=self.regularizer,
                                 trainable=True
                                 )
        self.bias = None
        self.built = True

    def call(self, inputs, **kwargs):  # 实现本层从输入张量到输出张量的计算图
        inputs = K.tf.nn.l2_normalize(inputs, 1, 1e-10)  # X 归一化
        self.W = K.tf.nn.l2_normalize(self.W, 0, 1e-10)  # W 归一化
        # cos(θ) --------------------------------------------------------------
        cos_theta = K.dot(inputs, self.W)
        # CosFace ====================== 余弦距离 =====================
        # phi = cos_theta - self.m
        # ArcFace ====================== 角度距离 =====================
        # controls the (theta + m) should in range [0, pi]
        theta = K.tf.acos(K.clip(cos_theta, -1.0+K.epsilon(), 1.0-K.epsilon()))
        phi = K.tf.cos(theta + self.m)
        # e^φ -----------------------------------------------------------------
        e_phi = K.exp(self.s * phi)
        e_cos = K.exp(self.s * cos_theta)
        # output
        output = e_phi / (e_phi + (K.sum(e_cos, axis=-1, keepdims=True)-e_cos))
        return output

    def compute_output_shape(self, input_shape):  # 指定输入及输出张量形状变化的逻辑!
        return input_shape[0], self.output_dim


# Loss
def loss_defined(y_true, y_pred):
    loss = -K.mean(K.log(K.clip(K.sum(y_true * y_pred, axis=-1), K.epsilon(), None)), axis=-1)
    return loss


# model
def build_net():
    model = Sequential([InputLayer(input_shape=(28, 28, 1)),

                        Conv2D(32, (3, 3), activation='relu'),
                        Conv2D(32, (3, 3), activation='relu'),
                        BatchNormalization(),
                        MaxPooling2D(pool_size=(2, 2)),

                        Conv2D(64, (3, 3), activation='relu'),
                        Conv2D(64, (3, 3), activation='relu'),
                        BatchNormalization(),
                        MaxPooling2D(pool_size=(2, 2)),

                        Flatten(),
                        Dropout(0.25),
                        Dense(128, activation='relu'),

                        Dense(2, name="feature_embedding"),
                        ArcLayer(10, s=30.0, m=0.5)  # CosFace: s=24, m=0.2
                        ])
    # -------------------------------------
    model.summary()
    model.compile(loss=loss_defined, optimizer=Adam(), metrics=['accuracy'])
    return model


if __name__ == "__main__":
    model = build_net()
    plot_model(model, to_file="./model.png", show_shapes=True)
