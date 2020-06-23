## Given a picture and its face landmarks, it should put 4 on the center line.
## then rotate the picture along 4 so that 3 and 1 are horizontal
import face_detector
import cv2
import numpy as np

IDEAL_IMG = cv2.imread('ideal.png')
IDEAL_FACE = face_detector.get_face(IDEAL_IMG)

EXPECTED_LOCATIONS = face_detector.get_landmarks(IDEAL_IMG,IDEAL_FACE)
IMPORTANT_POINTS = (36,45,8)

def get_fixed_image(image,landmarks):
    rows,cols,ch = image.shape
    M = cv2.getAffineTransform(np.float32(landmarks[IMPORTANT_POINTS,:]),np.float32(EXPECTED_LOCATIONS[IMPORTANT_POINTS,:]))
    ## We want a distance between eyes equal to 50 pixels (arbitrary)
    dst = cv2.warpAffine(image,M,(rows,cols))

    return dst
