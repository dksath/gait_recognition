import cv2
import numpy as np
import os

# Playing video from file:
cap = cv2.VideoCapture('walking_sil.avi')

try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print ('Error: Creating directory of data')

currentFrame = 0
count = 0
save = 0

ret, frame = cap.read()

while ret and save < 80 :
    # Capture frame-by-frame

    ret, frame = cap.read()
    # Saves image of the current frame in jpg file
    name = './data/frame' + str(currentFrame) + '.png'
    print ('Creating...' + name)
    cv2.imwrite(name, frame)


    ret, frame = cap.read()

    # To stop duplicate images
    currentFrame += 1
    count+=1
    save+=1
    

# When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()