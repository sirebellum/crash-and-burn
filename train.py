import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras import callbacks
from tensorflow.keras import datasets, layers, models, regularizers
from tensorflow.keras import Input, Model

from sklearn.metrics import classification_report

import numpy as np
import cv2

from dataset import QRDataset
from multiprocessing import Pool

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def make_dataset(size=1000):
    return QRDataset(n_samples=size)

if __name__ == "__main__":
    with tf.device('/gpu:1'):

        batches = 2
        threads = 32
        with Pool(threads) as p:
            data = p.map(make_dataset, [1000]*threads*batches)
        trains = data[:30*batches]
        vals = data[30*batches:]

        # Process stacks
        print("Stackin")
        train_data, train_sects, train_rots = [], [], []
        train_onehot = []
        for train in trains:
            train_data.append(train.images)
            train_sects.append(train.sections)
            train_rots.append(train.rotations)
            train_onehot.append(train.onehot)
        train_data = np.concatenate(train_data, axis=0).astype("float16")
        train_sects = np.concatenate(train_sects, axis=0).astype("uint8")
        train_rots = np.concatenate(train_rots, axis=0).astype("float16")
        train_onehot = np.concatenate(train_onehot, axis=0).astype("bool")

        # ... again
        print("Stackin")
        val_data, val_sects, val_rots = [], [], []
        val_onehot = []
        for val in vals:
            val_data.append(val.images)
            val_sects.append(val.sections)
            val_rots.append(val.rotations)
            val_onehot.append(val.onehot)
        val_data = np.concatenate(val_data, axis=0).astype("float16")
        val_sects = np.concatenate(val_sects, axis=0).astype("uint8")
        val_rots = np.concatenate(val_rots, axis=0).astype("float16")
        val_onehot = np.concatenate(val_onehot, axis=0).astype("bool")

        conv_reg = regularizers.L1L2(l1=1e-4, l2=1e-4)

        inputs = Input(shape=val_data[0].shape, name="input")

        # Common
        x = layers.Conv2D(8, (3, 3),
                        padding="valid",
                        activation='linear',
                        kernel_regularizer=conv_reg,)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Conv2D(16, (3, 3),
                        padding="valid",
                        activation='linear',
                        kernel_regularizer=conv_reg,)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Conv2D(32, (3, 3),
                        padding="valid",
                        activation='linear',
                        kernel_regularizer=conv_reg,)(x)
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

        # Localization
        x0 = layers.Conv2D(64, (3, 3),
                        padding="valid",
                        activation='linear',
                        kernel_regularizer=conv_reg)(x)
        x0 = layers.BatchNormalization()(x0)
        x0 = layers.ReLU()(x0)
        x0 = layers.MaxPooling2D((2,2))(x0)
        x0 = layers.Flatten()(x0)

        output0 = layers.Dense(32, activation="relu",
                        kernel_regularizer=conv_reg,)(x0)
        output0 = layers.Dropout(0.3)(output0)
        output0 = layers.Dense(9,
                            activation="softmax",
                            name="section_out")(output0)

        # Rotation
        x1 = layers.Conv2D(64, (3, 3),
                        padding="valid",
                        activation='linear',
                        kernel_regularizer=conv_reg)(x)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.ReLU()(x1)
        x1 = layers.Conv2D(128, (3, 3),
                        padding="valid",
                        activation='linear',
                        kernel_regularizer=conv_reg)(x1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.ReLU()(x1)
        x1 = layers.MaxPooling2D((2,2))(x1)
        x1 = layers.Flatten()(x1)

        output1 = layers.Dense(32, activation="relu",
                        kernel_regularizer=conv_reg,)(x1)
        output1 = layers.Dropout(0.3)(output1)
        output1 = layers.Dense(2,
                            activation="softmax",
                            name="rot_out")(output1)

        model = Model(inputs=inputs, outputs=[output0, output1])
        model.summary()

        model.compile(optimizer='adam',
                      loss={'section_out': 'categorical_crossentropy',
                            'rot_out': 'mse'},
                      metrics={'section_out': 'categorical_accuracy'})

        model.fit(train_data, [train_onehot, train_rots],
                  batch_size=128,
                  validation_data=(val_data, [val_onehot, val_rots]),
                  epochs=20)
        model.save("cab")

    input("Press enter to continue with demo")
    for image, label in zip(val.images, val.sections):
        sect, rot = model.predict(np.expand_dims(image, axis=0))
        rev_rot = int(-rot[0][0]*360)
        pred_label = np.argmax(sect[0])
        image = (image*255).astype("uint8")
        cv2.imshow(f"{pred_label}", cv2.resize(image, (128,128)))
        cv2.imshow(f"rotated", rotate_image(cv2.resize(image, (128,128)), rev_rot))
        cv2.waitKey(500)
