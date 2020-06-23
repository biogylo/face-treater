from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

import face_treater_config as cfg
import extra_functions

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(cfg.FACE_SHAPE_PREDICTOR_LOCATION)

def get_blurryness(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def get_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def get_face(image): ## gets the bounding box for rectangles
    ## Returns a dlib rectangle, or false if no face was found
    faces = detector(get_gray(image))

    if len(faces) == 1:
        return faces[0]
    else:
        return len(faces)
def get_landmarks(image,face):
    return face_utils.shape_to_np(predictor(get_gray(image),face))
