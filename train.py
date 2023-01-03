import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras import callbacks
from tensorflow.keras import datasets, layers, models, regularizers
from tensorflow.keras import Input, Model

from sklearn.metrics import classification_report
from datetime import datetime
import numpy as np
import cv2

from dataset import QRDataset
from multiprocessing import Pool

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

if __name__ == "__main__":
    with tf.device('/gpu:1'):

        train = QRDataset(n_samples=200000)
        val = QRDataset(n_samples=20000)

        conv_reg = regularizers.L1L2(l1=1e-5, l2=1e-5)
        dense_reg = regularizers.L1L2(l1=1e-5, l2=1e-5)

        inputs = Input(shape=val.image_size, name="input")

        # Common
        x = layers.Conv2D(32, (3, 3),
                        padding="valid",
                        activation='linear',
                        kernel_regularizer=conv_reg,)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Conv2D(64, (3, 3),
                        padding="valid",
                        activation='linear',
                        kernel_regularizer=conv_reg,)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2,2))(x)

        x = layers.Conv2D(128, (3, 3),
                        padding="valid",
                        activation='linear',
                        kernel_regularizer=conv_reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Conv2D(128, (3, 3),
                        padding="valid",
                        activation='linear',
                        kernel_regularizer=conv_reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Conv2D(128, (3, 3),
                        padding="valid",
                        activation='linear',
                        kernel_regularizer=conv_reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Conv2D(128, (3, 3),
                        padding="valid",
                        activation='linear',
                        kernel_regularizer=conv_reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2,2))(x)
        x_flat = layers.Flatten()(x)

        # Localization
        output0 = layers.Dense(128,
                               kernel_regularizer=dense_reg,
                               activation="relu")(x_flat)
        output0 = layers.Dense(val.n_sections,
                            kernel_regularizer=dense_reg,
                            activation="softmax",
                            name="section_out")(output0)

        # Rotation
        output1 = layers.Dense(128,
                               kernel_regularizer=dense_reg,
                               activation="relu")(x_flat)
        output1 = layers.Dense(2,
                            kernel_regularizer=dense_reg,
                            activation="softmax",
                            name="rot_out")(output1)

        model = Model(inputs=inputs, outputs=[output0, output1])
        model.summary()

        model.compile(optimizer='adam',
                      loss={'section_out': 'categorical_crossentropy',
                            'rot_out': 'mae'},
                      metrics={'section_out': 'categorical_accuracy'})

        log_dir = 'tensorboard/'+datetime.now().strftime("%d/%m-%H:%M")
        callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir)]
        model.fit(train.images, [train.onehot, train.rotations],
                  batch_size=256,
                  validation_data=(val.images, [val.onehot, val.rotations]),
                  epochs=25,
                  callbacks=callbacks)
        model.save("cab")

    # input("Press enter to continue with demo")
    # for image, label in zip(val.images, val.sections):
    #     sect, rot = model.predict(np.expand_dims(image, axis=0))
    #     rev_rot = int(-rot[0][0]*360)
    #     pred_label = np.argmax(sect[0])
    #     image = (image*255).astype("uint8")
    #     cv2.imshow(f"{pred_label}", cv2.resize(image, (128,128)))
    #     cv2.imshow(f"rotated", rotate_image(cv2.resize(image, (128,128)), rev_rot))
    #     cv2.waitKey(500)
