# import image dependencies (OpenCV)
import preprocessing
import loadfiles
import cv2
import csv
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Conv2D, Flatten, Dropout
from keras.applications import ResNet50
# experimental_list_devices depriciated in tensorflow2.1
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback
import matplotlib.pyplot as plt
import random

def printImage(img):
    cv2.imshow('Original Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def findMax(arr,dict):
    # find largest element in array and return breed the element corresponds to
    maxValue = -1
    maxIndex = 0
    for index in range(len(arr)):
        if arr[index] > maxValue:
            maxValue = arr[index]
            maxIndex = index
    key_list = list(dict.keys())
    val_list = list(dict.values())
    return key_list[val_list.index(maxIndex)]

def loadBModel():
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_size, img_size, channels)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.25))

# code injection - experimental_list_devices depricaited in tensorflow2.1
def _get_available_gpus():
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

# Assign label precentages to test images
tfback._get_available_gpus = _get_available_gpus #experimental_list_devices depriciated in tf2.1

# load images and label into array
num_classes = 120
img_size = 32
channels = 3
epochs = 2#7
(npTrainImages, npTrainLabels, categoricalDict) = preprocessing.processData(img_size, channels)
# create model
model = Sequential()
# instantiates the ResNet50V2 architecture
model.add(ResNet50(include_top = False, pooling = "avg", weights = "imagenet"))
#loadBModel()
# load weights into model -number of layers must match
#model.load_weights("catdog_weight.h5")
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# train the model - pass in numpy array of training data
history = model.fit(npTrainImages, npTrainLabels, validation_split=0.2, batch_size=32, epochs=epochs, verbose=1)
# save model weights - pip install h5py
# model.save_weights("catdog_weight.h5")

# predict using unseen data
testImage = []
testNames = []
predictBreed = []
loadfiles.loadImage(testImage, testNames, "test", img_size, 3)
npTestImages = np.asarray(testImage)
npTestImages = npTestImages.reshape(npTestImages.shape[0], img_size, img_size, channels).astype('float32')
# normalize data
npTestImages = npTestImages/255 + 0.01
predictList = model.predict(npTestImages[:])
for index in range(len(predictList)):
    breedList = []
    for breed in predictList[index]:
        breedList.append(breed)
    predictBreed.append(findMax(breedList, categoricalDict))

#  write predictions to csv file
with open('predictions.csv', 'w', newline='') as csv_file:
    fileWriter = csv.writer(csv_file)
    for index in range(len(predictList)):
        fileWriter.writerow([testNames[index], predictBreed[index]])

# plot loss history
plt.plot(history.history['val_loss'])
plt.title('Validation loss history')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.show()
# Plot accuracy history
plt.plot(history.history['val_accuracy'])
plt.title('Validation accuracy history')
plt.ylabel('Accuracy value (%)')
plt.xlabel('No. epoch')
plt.show()