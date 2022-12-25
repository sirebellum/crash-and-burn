import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras import datasets, layers, models, regularizers

from sklearn.metrics import classification_report

import numpy as np
import cv2

from dataset import QRDataset

with tf.device('/gpu:1'):
    train = QRDataset(n_samples=10000)
    val = QRDataset(n_samples=1000)

    model = models.Sequential()
    conv_reg = regularizers.L1L2(l1=1e-5, l2=1e-4)
    model.add(layers.Conv2D(8, (3, 3),
                            padding="valid",
                            activation='linear',
                            kernel_regularizer=conv_reg,
                            input_shape=val.image_size))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3),
                            padding="valid",
                            activation='linear',
                            kernel_regularizer=conv_reg))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(16, (3, 3),
                            padding="valid",
                            activation='linear',
                            kernel_regularizer=conv_reg))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3),
                            padding="valid",
                            activation='linear',
                            kernel_regularizer=conv_reg))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(32, (3, 3),
                            padding="valid",
                            activation='linear',
                            kernel_regularizer=conv_reg))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3),
                            padding="valid",
                            activation='linear',
                            kernel_regularizer=conv_reg))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Flatten())
    model.add(layers.Dense(val.n_sections,
                           activation="softmax"))
    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.fit(train.images, train.onehot,
              batch_size=128,
              validation_data=(val.images, val.onehot),
              epochs=20)
    print(classification_report(val.sections, np.argmax(model.predict(val.images), axis=1)))

    model.save("cab")

for image, label in zip(val.images, val.sections):
    pred_label = np.argmax(model.predict(np.expand_dims(image, axis=0)))
    cv2.imshow(f"{pred_label}", cv2.resize(image, (128,128)))
    cv2.waitKey(100)
