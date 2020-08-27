import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters
from skimage.transform import resize, rotate
import PIL
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, smart_resize
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc


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
                optimizer=Adam(), 
                metrics=['accuracy'])
    return model



if __name__=='__main__':
    img_width, img_height = 128, 128 
    train_data_dir = 'data/wave/training'
    validation_data_dir = 'data/wave/testing'
    epochs = 400
    batch_size = 24

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = cnn_model() 

    wave_train_datagen = ImageDataGenerator(
        zoom_range=0.2,
        height_shift_range=0.1,
        rescale=1./255,
        shear_range=0.2,
        horizontal_flip=True,
        ) 

    # this is the augmentation configuration for testing:
    wave_test_datagen = ImageDataGenerator(rescale=1./255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = wave_train_datagen.flow_from_directory(
            train_data_dir, 
            target_size=(img_width, img_height),
            batch_size=24,
            class_mode='binary',
            shuffle = True
            ) 

    # this is a similar generator, for validation data
    validation_generator = wave_test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=30,
            class_mode='binary',
            shuffle = False)

    # model.summary() 
    
    # history = model.fit(
    #             train_generator,
    #             steps_per_epoch=3,
    #             epochs=epochs,
    #             validation_data=validation_generator,
    #             validation_steps=1)

    model.evaluate(validation_generator)

    STEP_SIZE_TEST = validation_generator.n//validation_generator.batch_size
    validation_generator.reset()
    x, classes = next(validation_generator)
    preds = model.predict(x, verbose=1)

    fpr, tpr, _ = roc_curve(classes,preds)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkturquoise',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Wave Model Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # print(model.predict(validation_generator))

    # plt.plot(history.history['accuracy'], label='accuracy')
    # plt.plot(history.history['val_accuracy'], label= 'val_accuracy')
    # plt.plot(history.history['loss'], label='loss')
    # plt.plot(history.history['val_loss'], label='val_loss')
    # plt.xlabel('Epoch')
    # # plt.ylabel('Accuracy')
    # plt.ylim([0,1])
    # plt.legend(loc='lower right')
    # plt.show()
    # # plt.savefig('1000.png')

    # model.save_weights('try.h5') 