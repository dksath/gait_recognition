##################################################
# Instance Segmentation using Deeplabv3
##################################################
# It uses cv2 to start a video capture and 
# renders the output which is either 
# blurred or background substituted
##################################################
# Author: Vinayak Nayak
# Date: 2020-12-13
# site: https://towardsdatascience.com/semantic-image-segmentation-with-deeplabv3-pytorch-989319a9a4fb
# github: https://github.com/ElisonSherton/instanceSegmentation
##################################################
# Edited for Human Silhouette Segmentation
# Date: 2022-03-25

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import time


# Load the DeepLabv3 model to memory
model = utils.load_model()

# Evaluate model
model.eval()
print("Model has been loaded.")

prev_frame_time = 0
new_frame_time = 0

# Start a video cam session
video_session = cv2.VideoCapture(0)

# Define two axes for showing the mask and the true video in realtime
# And set the ticks to none for both the axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 8))

ax1.set_title("Original")
ax2.set_title("Mask")

ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])

# Create two image objects to picture on top of the axes defined above
im1 = ax1.imshow(utils.grab_frame(video_session))
im2 = ax2.imshow(utils.grab_frame(video_session))

# Switch on the interactive mode in matplotlib
plt.ion()
plt.show()

# Read frames from the video, make realtime predictions and display the same
while True:
    frame = utils.grab_frame(video_session)

    # Ensure there's something in the image (not completely blank)
    if np.any(frame):

        # Read the frame's width, height, channels and get the labels' predictions from utilities
        width, height, channels = frame.shape
        labels = utils.get_pred(frame, model)

        mask = labels == 15
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis = 2)
        
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time 
        print("fps: {}".format(str(fps)))
        fig.suptitle("fps: {}".format(str(fps)))

        # Set frames for output
        im1.set_data(frame)
        im2.set_data(mask * 255)
        plt.pause(0.01)

    else:
        break

# Empty the cache and switch off the interactive mode
torch.cuda.empty_cache()
plt.ioff()
