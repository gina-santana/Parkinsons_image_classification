import matplotlib.pyplot as plt
from skimage import io, color, filters
from skimage.transform import resize, rotate
import PIL
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc


def cnn_model():
    if K.image_data_format() == 'channels_first':
        input_shape = (3, model_params['img_width'], model_params['img_height'])
    else:
        input_shape = (model_params['img_width'], model_params['img_height'], 3)

    model = Sequential()

    for i, num_filters in enumerate(model_params['filters']):
        if i == 0:
            model.add(Conv2D(num_filters, (3, 3), input_shape=input_shape))
        else:
            model.add(Conv2D(num_filters, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  
    model.add(Dense(500)) 
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1)) # single output neuron (output ranged from 0-1; binary class)
    model.add(Activation('sigmoid'))
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(),  
        metrics=['accuracy']
    )

    return model


def data_augmentation():
    train_datagen = ImageDataGenerator(
        zoom_range = model_params['zoom_range'],
        height_shift_range = 0.1,
        rotation_range =  model_params['rotation_range'],
        rescale = 1./255,
        shear_range = 0.2,
        horizontal_flip = model_params['horizontal_flip']
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            model_params['train_data_dir'],  
            target_size=(model_params['img_width'], model_params['img_height']),
            batch_size=24,
            class_mode='binary',
            shuffle = True
        )

    validation_generator = validation_datagen.flow_from_directory(
            model_params['validation_data_dir'],
            target_size=(model_params['img_width'], model_params['img_height']),
            batch_size=30,
            class_mode='binary',
            shuffle = True # Change to False if using plot_roc
        )

    return (train_datagen, validation_datagen, train_generator, validation_generator)


def plot_roc():
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
    plt.title('Spiral Model Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def model_evaluation_plot():
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label= 'val_accuracy')
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylim([0,1])
    plt.legend(loc='lower right')
    plt.show()


if __name__=='__main__':

    spiral_params = {
        'filters': [32, 32, 32, 32, 64],
        'img_width': 256,
        'img_height': 256,
        'train_data_dir': 'data/spiral/training',
        'validation_data_dir': 'data/spiral/testing',
        'epochs': 400,
        'batch_size': 24,
        'zoom_range': 0.1,
        'rotatation_range': 360,
        'horizontal_flip': False,
        'weights': 'src/spiral.h5'
    }

    wave_params = {
        'filters': [32, 32, 32, 64],
        'img_width': 128,
        'img_height': 128,
        'train_data_dir': 'data/wave/training',
        'validation_data_dir': 'data/wave/testing',
        'epochs': 750,
        'batch_size': 24,
        'zoom_range': 0.2,
        'rotation_range': 0,
        'horizontal_flip': True,
        'weights': 'src/wave.h5'
    }

    model_params = wave_params #'spiral_params'
    
    model = cnn_model()

    train_datagen, validation_datagen, train_generator, validation_generator = data_augmentation()

    # model.load_weights(model_params['weights'])

    model.summary() 

    history = model.fit(
        train_generator,
        steps_per_epoch = 3,
        epochs = model_params['epochs'],
        validation_data = validation_generator,
        validation_steps = 1
    )

    model.evaluate(validation_generator)

    # plot_roc() # Change validation shuffle to False if using plot_roc 

    model_evaluation_plot()

    # model.save_weights('spiral_2.h5') 
