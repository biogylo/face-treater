
import argparse
import numpy as np
import cv2
import face_treater_config as cfg

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye.xml")

def get_blurryness(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def get_landmarks(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    return faces

img = cv2.imread("raw_pictures/billie-eilish-1.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
for (x,y,w,h) in get_landmarks(img):
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)

for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
