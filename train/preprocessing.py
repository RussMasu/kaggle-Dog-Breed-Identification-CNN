import loadfiles
import random
import numpy as np


def randArr(arr):
    # returns arr with randomized elements
    list = random.shuffle(arr)
    return list

def categoricalList(inlist,outlist):
    dic = {}
    i = 0
    for item in inlist:  # note O(n)=list(n)dict(n)
        if item not in dic.keys():
            dic.update({item:i})
            i=i+1
    for item in inlist:
        outlist.append(dic.get(item))

def processData(img_size,channels):
    #returns npTrainImages,npTrainLabels tuple
    # shuffling of data is needed if ordered since model trains in batches
    data = []  # list of tuples

    loadfiles.loadData(data, "labels.csv", "train", img_size, channels)
    randArr(data)

    trainImages = []
    trainLabels = []
    for i in range(len(data)):
        (img, lab) = data[i]
        trainImages.append(img)
        trainLabels.append(lab)

    # convert list to np.array (much faster then appending to np.array)
    npTrainImages = np.asarray(trainImages)
    # preprocessing - reshape data to fit keras model
    npTrainImages = npTrainImages.reshape(npTrainImages.shape[0], img_size, img_size, channels).astype('float32')
    # normalize data
    npTrainImages = npTrainImages / 255 + 0.01
    # preprocessing - string labels must be numeric
    npTrainLabels = []
    # create dictionary converting labels to int
    categoricalDict = {}
    i = 0
    for item in trainLabels:  # note O(n)=list(n)dict(n)
        if item not in categoricalDict.keys():
            categoricalDict.update({item: i})
            i = i + 1
        npTrainLabels.append(categoricalDict.get(item))

    # needed for categorical cross entropy/not sparse categorical cross entropy
    # npTrainLabels = to_categorical(trainCat)
    ans = (npTrainImages, npTrainLabels, categoricalDict)
    return ans
