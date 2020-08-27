import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters
from skimage.transform import resize, rotate
import PIL
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K


def cnn_model():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))  
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3)))  
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3))) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  
    model.add(Dense(500)) 
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1)) # single output neuron (output ranged from 0-1; binary class)
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                optimizer='adam',  
                metrics=['accuracy'])
    return model


# def plot_augmentation():
#     for i in range(9):
#         plt.subplot(300 + 1 + i)
#         batch = train_generator.next()
#         image = batch[0].astype('uint8')
#         plt.imshow(image)
#     return plt.show()

if __name__=='__main__':
    img_width, img_height = 256, 256
    train_data_dir = 'data/spiral/training'
    validation_data_dir = 'data/spiral/testing'
    epochs = 275
    batch_size = 24

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = cnn_model() 

    spiral_train_datagen = ImageDataGenerator(
        zoom_range=0.1,
        height_shift_range=0.1,
        rotation_range = 360, #spiral images 
        rescale=1./255,
        shear_range=0.2,
        horizontal_flip=False)

    # this is the augmentation configuration for testing:
    spiral_test_datagen = ImageDataGenerator(rescale=1./255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = spiral_train_datagen.flow_from_directory(
            train_data_dir,  
            target_size=(img_width, img_height),
            batch_size=24,
            class_mode='binary',
            shuffle = True
            ) 

    # validation data image generator
    validation_generator = spiral_test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=30,
            class_mode='binary',
            shuffle = True)

    model.summary() 
    
    history = model.fit(
                train_generator,
                steps_per_epoch=3,
                epochs=epochs,
                validation_data=validation_generator,
                validation_steps=1)

    print(model.predict(validation_generator))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label= 'val_accuracy')
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0,1])
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig('275V2.png')

    # model.save_weights('spiral.h5') 