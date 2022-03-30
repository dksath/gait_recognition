# importing libraries
import numpy as np
import cv2
import time

# creating object
fgbg1 = cv2.bgsegm.createBackgroundSubtractorMOG();
fgbg2 = cv2.createBackgroundSubtractorMOG2();
fgbg3 = cv2.bgsegm.createBackgroundSubtractorGMG();

prev_frame_time = 0
new_frame_time = 0

# capture frames from a camera
cap = cv2.VideoCapture(0);
while(1):
	# read frames
	ret, img = cap.read();
	
	# apply mask for background subtraction
	fgmask1 = fgbg1.apply(img);
	fgmask2 = fgbg2.apply(img);
	fgmask3 = fgbg3.apply(img);
	
	cv2.imshow('Original', img);
	cv2.imshow('MOG', fgmask1);
	cv2.imshow('MOG2', fgmask2);
	cv2.imshow('GMG', fgmask3);

	new_frame_time = time.time()
	fps = 1/(new_frame_time-prev_frame_time)
	prev_frame_time = new_frame_time 
	print("fps: {}".format(str(fps)))

	k = cv2.waitKey(30) & 0xff;
	if k == 27:
		break;

cap.release();
cv2.destroyAllWindows();
