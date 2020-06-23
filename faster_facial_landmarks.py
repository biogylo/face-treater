##code from https://www.pyimagesearch.com/2018/04/02/faster-facial-landmark-detector-with-dlib/
# import the necessary packages
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2

# construct the argument parser and parse the arguments
shape_predictor = 'dlib/shape_predictor_5_face_landmarks.dat'
image_filename = 'raw_pictures/billie-eilish-1.jpg'

# initialize dlib's face detector (HOG-based) and then create the
# facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)
# initialize the video stream and sleep for a bit, allowing the
# camera sensor to warm up

img = cv2.imread(image_filename)

# vs = VideoStream(usePiCamera=True).start() # Raspberry Pi
time.sleep(2.0)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale img
rects = detector(gray, 0)
# check to see if a face was detected, and if so, draw the total
# number of faces on the img
if len(rects) > 0:
	text = "{} face(s) found".format(len(rects))
	cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
		0.5, (0, 0, 255), 2)

for rect in rects:
		# compute the bounding box of the face and draw it on the
		# img
		(bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
		cv2.rectangle(img, (bX, bY), (bX + bW, bY + bH),
			(0, 255, 0), 1)
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw each of them
		for (i, (x, y)) in enumerate(shape):
			cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
			cv2.putText(img, str(i), (x - 10, y - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
cv2.imshow('Wait', img)
cv2.waitKey(0)
