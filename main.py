#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from os import listdir
from os.path import isfile, join

from imutils import face_utils
import numpy as np
import imutils
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./model/shape_predictor_68_face_landmarks.dat")

def extractImages(pathIn, pathOut, msec):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*msec))    # added this line
        success,image = vidcap.read()
        if not success :
            break
        cv2.imwrite( pathOut + "frame%03d.jpg" % count, image)     # save frame as JPEG file
        count = count + 1

def countFaces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    return len(rects)

MUSICBANK_START = 64
MCOUNTDOWN_START = 21

def createCandidate():
    mb_frames = [join("./musicbank_images",f) for f in listdir("./musicbank_images") if isfile(join("./musicbank_images", f))]
    mc_frames = [join("./mcountdown_images",f) for f in listdir("./mcountdown_images") if isfile(join("./mcountdown_images", f))]

    mb_frames.sort()
    mc_frames.sort()

    mbi = MUSICBANK_START
    mci = MCOUNTDOWN_START
    while True:
        mbf = mb_frames[mbi]
        mcf = mc_frames[mci]
        mbf_img = imutils.resize(cv2.imread(mbf), width=500)
        mcf_img = imutils.resize(cv2.imread(mcf), width=500)
        
        mbf_face_cnt = countFaces(mbf_img)
        mcf_face_cnt = countFaces(mcf_img)
        if mbf_face_cnt == mcf_face_cnt == 1:
            print("Checkout mbf %d and mcf %d"%(mbi, mci))
            cv2.imshow("Musicbank", mbf_img)
            cv2.imshow("Mcountdown", mcf_img)
            cv2.waitKey(0)
        mbi += 1
        mci += 1

if __name__ == "__main__":    
    print("welcome to x-editing")

    ### Extract 1 frame per sec
    extractImages("./videos/musicbank.mp4", "./musicbank_images/", 1000)
    extractImages("./videos/mcountdown.mp4", "./mcountdown_images/", 1000)

    createCandidate()

