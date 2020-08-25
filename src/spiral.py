import numpy as np
from skimage import io, color, filters
from skimage.transform import resize, rotate
import PIL
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.nn import leaky_relu

img_width, img_height = 256, 256
train_data_dir = 'data/spiral/training'
validation_data_dir = 'data/spiral/testing'
# nb_train_samples = 100
# nb_validation_samples = 100
epochs = 10
batch_size = 36

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25)) 

model.add(Conv2D(32, (3, 3))) # added layer
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25)) 

model.add(Conv2D(64, (3, 3))) # added layer
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25)) 

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(1000)) # originally 64 changed to 500
model.add(Activation('relu'))
# model.add(Dropout(0.5)) 
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',  # changed to adam
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
spiral_train_datagen = ImageDataGenerator(
        zoom_range=0.1,
        height_shift_range=0.1,
        rotation_range = 360, #spiral images 
        rescale=1./255,
        shear_range=0.2,
        horizontal_flip=False)


# this is the augmentation configuration we will use for testing:
# only rescaling
spiral_test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = spiral_train_datagen.flow_from_directory(
        train_data_dir,  # this is the target directory
        target_size=(img_width, img_height),
        batch_size=35,
        class_mode='binary',
        seed = 4,
        shuffle = True,
        )  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = spiral_test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=30,
        class_mode='binary',
        seed = 4,
        shuffle = True,)

model.fit(
        train_generator,
        steps_per_epoch=2,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=1)
model.save_weights('first_try.h5')  # always save your weights after training or during training

model.summary() 

