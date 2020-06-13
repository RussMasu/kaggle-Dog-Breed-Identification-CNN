import os
import glob
import cv2
import csv
import re

def parseImageLable(img):
    lst = re.findall("[^\\\]*$", img)
    lst2 = re.findall("(.*)\.[^.]+$", lst[0])
    return lst2[0]


def loadData(arr, labelpath, path, imsize,chan):
    labelarr = []
    imgarr = []
    farr = []

    loadImage(imgarr, farr, path, imsize, chan)
    loadLabels(labelarr, labelpath)
    for i in range(len(imgarr)):
        arr.append((imgarr[i], labelarr[i]))

def loadImage(imgarr, farr, path, imsize,chan):
    img_dir = path
    data_path = os.path.join(img_dir, '*g')
    files = glob.glob(data_path)

    #load image data
    for file in files:
        name = parseImageLable(file)
        img = cv2.imread(file)
        # resize img
        img = cv2.resize(img, (imsize, imsize))
        # convert image to gray scale
        if chan == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgarr.append(img)
        farr.append(name)
        # flip image
        # img = cv2.flip(img, 1)
        # imgarr.append(img)

def loadLabels(labelarr, fname):
    with open(fname, newline='') as csvfile:
        file = csv.reader(csvfile)
        next(file)  # skip the first line
        for row in file:
            (first, second) = row
            labelarr.append(second)