# ========================================================================================
# PROBLEM B3
#
# Build a CNN based classifier for Rock-Paper-Scissors dataset.
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# This is unlabeled data, use ImageDataGenerator to automatically label it.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
#
# Desired accuracy AND validation_accuracy > 83%
# ========================================================================================

import urllib.request
from keras import Model
import tensorflow as tf
from keras import layers
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator


class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > .72 and logs.get('val_accuracy') > .72:
            self.model.stop_training = True

def emotion_detection():
    TRAINING_DIR = "images/train"
    training_datagen = ImageDataGenerator(
        rescale=1/ 255
    )

    VALIDATION_DIR = "images/validation"
    validation_datagen = ImageDataGenerator(
        rescale=1/ 255
    )

    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(48, 48),
        class_mode='categorical',
        color_mode='grayscale',
        batch_size=64
    )

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(48, 48),
        class_mode='categorical',
        color_mode='grayscale',
        batch_size=64
    )
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(7, activation='softmax')
    ])


    model.compile(loss = 'categorical_crossentropy', optimizer=Adam(learning_rate=0.001, decay=1e-6), metrics=['accuracy'])
    # TRAIN YOUR MODEL HERE
    model.fit(train_generator,
              epochs=40,
              validation_data =validation_generator,
              verbose = 1,
              )

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model=emotion_detection()
    model.save("skripsi.h5")
