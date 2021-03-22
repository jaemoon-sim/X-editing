#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from os import listdir
from os.path import isfile, join

from imutils import face_utils
import numpy as np
import imutils
import dlib
import math

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

def detectFaceRects(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    return rects


MUSICBANK_START = 64
MCOUNTDOWN_START = 21

def createCandidate():
    mb_frames = [join("./musicbank_images",f) for f in listdir("./musicbank_images") if isfile(join("./musicbank_images", f))]
    mc_frames = [join("./mcountdown_images",f) for f in listdir("./mcountdown_images") if isfile(join("./mcountdown_images", f))]

    mb_frames.sort()
    mc_frames.sort()

    mbi = MUSICBANK_START
    mci = MCOUNTDOWN_START

    result = []
    while True:
        mbf = mb_frames[mbi]
        mcf = mc_frames[mci]
        mbf_img = imutils.resize(cv2.imread(mbf), width=500)
        mcf_img = imutils.resize(cv2.imread(mcf), width=500)
        
        mbf_face_cnt = countFaces(mbf_img)
        mcf_face_cnt = countFaces(mcf_img)
        if mbf_face_cnt == mcf_face_cnt == 1:
            result.append((mbf, mcf))
            print("Checkout mbf %d and mcf %d"%(mbi, mci))
            cv2.imshow("Musicbank", mbf_img)
            cv2.imshow("Mcountdown", mcf_img)
            cv2.waitKey(0)
        mbi += 1
        mci += 1
        if mbi >= len(mb_frames) or mci >= len(mc_frames):
            break
    return result

def calculateMatrix(src_frame, dst_frame):
    src_img = imutils.resize(cv2.imread(src_frame), width=500)
    dst_img = imutils.resize(cv2.imread(dst_frame), width=500)

    src_rect = detectFaceRects(src_img)[0]
    dst_rect = detectFaceRects(dst_img)[0]

    src_tl = src_rect.tl_corner()
    src_tr = src_rect.tr_corner()
    src_br = src_rect.br_corner()
    src_pts = np.float32([[src_tl.x, src_tl.y],[src_tr.x, src_tr.y],[src_br.x, src_br.y]])

    dst_tl = dst_rect.tl_corner()
    dst_tr = dst_rect.tr_corner()
    dst_br = dst_rect.br_corner()
    dst_pts = np.float32([[dst_tl.x, dst_tl.y],[dst_tr.x, dst_tr.y],[dst_br.x, dst_br.y]])
   
    return cv2.getAffineTransform(src_pts, dst_pts)


if __name__ == "__main__":    
    print("welcome to x-editing")

    ### Extract 1 frame per sec
    #  extractImages("./videos/musicbank.mp4", "./musicbank_images/", 1000)
    #  extractImages("./videos/mcountdown.mp4", "./mcountdown_images/", 1000)

    #  candidates = createCandidate()
    
    ## let's work with 119 & 76
    mb_frames = [join("./musicbank_images",f) for f in listdir("./musicbank_images") if isfile(join("./musicbank_images", f))]
    mc_frames = [join("./mcountdown_images",f) for f in listdir("./mcountdown_images") if isfile(join("./mcountdown_images", f))]

    mb_frames.sort()
    mc_frames.sort()
    cand0 = mb_frames[119]
    cand1 = mc_frames[76]

    M = calculateMatrix(cand0, cand1)

    src_img = imutils.resize(cv2.imread(cand0), width=500)
    dst_img = imutils.resize(cv2.imread(cand1), width=500)

    src_img_translated = cv2.warpAffine(src_img, M, (src_img.shape[1], src_img.shape[0]))
    
    cv2.imshow("src", src_img)
    cv2.imshow("translated", src_img_translated)
    cv2.imshow("dest", dst_img)
    cv2.waitKey(0)
