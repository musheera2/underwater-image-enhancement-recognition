import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten,MaxPooling2D,Dropout,BatchNormalization,MaxPool2D,GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from matplotlib.pyplot import imshow
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet import ResNet50
from tensorflow.keras.applications import DenseNet201
import warnings
warnings.filterwarnings("ignore")



def rescale_img(img_path):
    
    img = load_img(img_path)
    array = img_to_array(img)
    print(f"Image Size: {array.shape}")
    img = img.resize([128,128])
    img_array = img_to_array(img).reshape(-1,128,128,3)
    # img_array = np.array(img_array)
   
    print(f"Rescaled the image to --> {img_array.shape} ")

    return img_array


def VGG19_model():

    IMAGE_SIZE=[128,128]
    vgg19_model = VGG19(input_shape =IMAGE_SIZE + [3], weights='imagenet', include_top=False)

    for layer in vgg19_model.layers:
        layer.trainable = False

    x = Flatten()(vgg19_model.output)
    x= Dense(800,activation="relu")(x)
    x= Dense(128,activation="relu")(x)
    x= Dropout(0.5)(x)
    prediction_layer = Dense(19, activation='softmax')(x)


    ### create a model object
    VGG19_model = Model(inputs=vgg19_model.input, outputs=prediction_layer)
    VGG19_model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    VGG19_model.load_weights('../MODELS/VGG19_model_on_underwater.h5')

    return ("VGG19_model",VGG19_model)


def AlexNet_model():
    model = Sequential([
    Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(128,128,3)),
    BatchNormalization(),
    MaxPool2D(pool_size=(3,3), strides=(2,2)),
    Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPool2D(pool_size=(3,3), strides=(2,2)),
    Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    BatchNormalization(),
    Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    BatchNormalization(),
    Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPool2D(pool_size=(3,3), strides=(2,2)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(19, activation='softmax')])
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    model.load_weights('../MODELS/AlexNet_model_on_underwater.h5')

    return  ("AlexNet Model",model)

def InceptionV3_model():
    img_height,img_width = 128,128 
    num_classes = 19
    base_model = InceptionV3(weights= None, include_top=False, input_shape= (img_height,img_width,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(120, activation='relu')(x)
    predictions = Dense(num_classes, activation= 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.load_weights('../MODELS/InceptionV3_model_on_underwater.h5')

    return ("InceptionV3 Model",model)
        

def ResNet50_model():
    img_height,img_width = 128,128 
    num_classes = 19
    base_model = ResNet50(weights= None, include_top=False, input_shape= (img_height,img_width,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation= 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.load_weights('../MODELS/ResNet50_model_on_underwater.h5')

    return ("ResNet50 Model",model)


def DenseNet201_model():
    img_height,img_width = 128,128 
    num_classes = 19
    densenet201=DenseNet201(
        include_top=False,
        weights=None,
        input_shape=(img_height,img_width,3),
        classes=19,
    )
    x = densenet201.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation= 'softmax')(x)
    densemodel = Model(inputs = densenet201.input, outputs = predictions)
    densemodel.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    densemodel.load_weights('../MODELS/dense201_model_on_underwater.h5')

    return ("DenseNet201 Model",densemodel)



