from imutils import paths
import argparse
import cv2
import face_treater_config as cfg

def get_blurryness(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()
