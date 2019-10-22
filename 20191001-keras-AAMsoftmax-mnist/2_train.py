from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from aam_softmax_model import build_net


train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=5,
                                   width_shift_range=0.01,
                                   height_shift_range=0.01,
                                   fill_mode="constant",
                                   cval=int(0)
                                   )
validation_datagen = ImageDataGenerator(rescale=1. / 255)
# 配置训练所用数据
train_generator = train_datagen.flow_from_directory(directory="datasets/mnist_train",
                                                    target_size=(28, 28),
                                                    color_mode="grayscale",
                                                    batch_size=64,
                                                    class_mode="categorical"
                                                    )
validation_generator = validation_datagen.flow_from_directory(directory="datasets/mnist_validation",
                                                              target_size=(28, 28),
                                                              color_mode="grayscale",
                                                              batch_size=64,
                                                              class_mode="categorical"
                                                              )
# -------------------------------------------------------------------------
model = build_net()
checkpoint = ModelCheckpoint(filepath="./model.h5", monitor='val_loss', verbose=1, save_best_only=True)
model.fit_generator(train_generator,
                    steps_per_epoch=train_generator.samples // 64,
                    epochs=100,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples // 64,
                    callbacks=[checkpoint]
                    )
