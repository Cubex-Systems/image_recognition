from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os

train = ImageDataGenerator(rescale=1 / 255)
validation = ImageDataGenerator(rescale=1 / 255)

train_dataset = train.flow_from_directory("ML/training",
                                          target_size=(1280, 720),
                                          batch_size=1,
                                          class_mode='sparse')
validation_dataset = train.flow_from_directory("ML/validation",
                                          target_size=(1280, 720),
                                          batch_size=1,
                                          class_mode='sparse')

print(train_dataset.class_indices)

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(1280, 720, 3)),
                                    tf.keras.layers.MaxPool2D(2, 2),
                                    #
                                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2, 2),
                                    #
                                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2, 2),
                                    ##
                                    tf.keras.layers.Flatten(),
                                    ##
                                    tf.keras.layers.Dense(512, activation='relu'),
                                    ##
                                    tf.keras.layers.Dense(1, activation='sigmoid')

                                    ])

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001), #
              metrics=['accuracy'])

model_fit = model.fit(train_dataset,
                      steps_per_epoch=1,
                      epochs=1,
                      validation_data=validation_dataset)


dir_path = "ML/testing"

# for i in os.listdir(dir_path):
img = image.load_img(dir_path+'//'+'test-template-3.jpg',target_size=(1280,720))
	# plt.imshow(img)
	# plt.show()

X= image.img_to_array(img)
X= np.expand_dims(X,axis=0)

images = np.vstack([X])

val = model.predict(images)

print(val)

print(np.argmax(val))