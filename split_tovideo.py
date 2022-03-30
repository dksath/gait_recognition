import cv2
import numpy as np
import os
import time

start = time.time()

# Playing video from file:
cap = cv2.VideoCapture('walking.mp4')

# Splitting video into frames
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
currentFrame = 0
count = 0
totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
interval = int(totalFrames / 80)
print("Total Frames: {}".format(totalFrames))
print("Interval: {}".format(interval))

img_array = []
if cap.isOpened():
    print("Reading video..")
    currentFrame = 0
    while True:
        ret, frame = cap.read()

        if ret:
            while currentFrame < 80:
                # Capture frame-by-frame

                ret, frame = cap.read()
                # Saves image of the current frame in img_array
                img_array.append(frame)

                # To stop duplicate images
                currentFrame += 1
                count += interval
                successful = cap.set(cv2.CAP_PROP_POS_FRAMES, count)
                #print("cap set at {}: {}".format(count, successful))
        #cap.release()
        break

# Writing to video
print("Writing split video..")
out = cv2.VideoWriter('splitvideo.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (width, height))

for i in range(len(img_array)):
    out.write(img_array[i])

cap.release()
out.release()

cv2.destroyAllWindows()

end = time.time()
print("Time elapsed: {}s".format(end - start))