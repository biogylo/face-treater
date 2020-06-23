
import argparse
import numpy as np
import cv2
import face_treater_config as cfg

face_cascade = cv2.CascadeClassifier("../../data/haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("../../data/haarcascades/haarcascade_eye.xml")

def get_blurryness(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def get_landmarks(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    return faces

image = cv2.imread("raw_pictures/billie-eilish-1.jpg")

for (x,y,w,h) in get_landmarks(image):
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)

cv2.imshow("test", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
