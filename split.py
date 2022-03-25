import cv2
import numpy as np
import os

# Playing video from file:
cap = cv2.VideoCapture('walking_sil.avi')

try:
    if not os.path.exists('data'):
        print("Creating '/data' directory..")
        os.makedirs('data')
    else:
        # already exists, clear
        print("'/data' directory found, clearing..")
        for file in os.listdir('data'):
            os.remove(os.path.join('data', file))
        print("'/data' cleared")
except OSError:
    print('Error: Creating directory of data')

currentFrame = 0
count = 0
totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
interval = int(totalFrames / 80)
print("Total Frames: {}".format(totalFrames))
print("Interval: {}".format(interval))

if cap.isOpened():
    currentFrame = 0
    while True:
        ret, frame = cap.read()

        if ret:
            while currentFrame < 80:
                # Capture frame-by-frame

                ret, frame = cap.read()
                # Saves image of the current frame in jpg file
                name = './data/frame' + str(currentFrame) + '.png'
                print('Creating...' + name)
                cv2.imwrite(name, frame)

                # To stop duplicate images
                currentFrame += 1
                count += interval
                successful = cap.set(cv2.CAP_PROP_POS_FRAMES, count)
                #print("cap set at {}: {}".format(count, successful))
        cap.release()
        break
cv2.destroyAllWindows()
