from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import os

size = (48, 48)

train_data_dir = 'ML/train'
validation_data_dir = 'ML/validation'

train_datagen = ImageDataGenerator(
    rescale=1. / 255
)

validation_datagen = ImageDataGenerator(
    rescale=1. / 255
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=size,
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=size,
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

class_labels = ['template_1', 'template_2', 'template_3', 'template_4']

img, label = train_generator.__next__()


model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),activation='relu', input_shape=(48,48,1)))

model.add(Conv2D(64,kernel_size=(3,3),activation='relu', ))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))


model.add(Conv2D(128,kernel_size=(3,3),activation='relu', ))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(256,kernel_size=(3,3),activation='relu', ))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(4,activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_data_dir = 'ML/train'
validation_data_dir = 'ML/validation'

num_train_imgs = 0
for root,dirs,files in os.walk(train_data_dir):
    num_train_imgs += len(files)

num_test_imgs = 0
for root, dirs, files in os.walk(validation_data_dir):
    num_test_imgs += len(files)


print(num_train_imgs)

print(num_test_imgs)


epochs = 100

history = model.fit(
    train_generator,
    steps_per_epoch=1,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=1
)

model.save('model_file.h5')