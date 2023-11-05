import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sounddevice as sd
from scipy.io.wavfile import read
import os
import sys
from skimage import exposure, color, filters, measure
import cv2

VFILE = "projet/20231031_144438.mp4"

def get_frames(filename):
    # Create a VideoCapture object 'video' to open the video file for reading.
    video = cv2.VideoCapture(filename)

    # Use a while loop to continuously read frames from the video until it's open.
    while video.isOpened():
        # Read the next video frame.
        ret, frame = video.read()

        # If 'ret' is True, the frame was read successfully.
        if ret:
            # Yield the current frame to the caller. 'yield' turns this function into a generator,
            # returning one frame at a time, but preserving the function state for the next call.
            yield frame
        else:
            # If 'ret' is False, it means there are no more frames to read in the video,
            # so we break the loop and stop the generator.
            break

    # Release the VideoCapture object to free up resources.
    video.release()

    # After the loop ends, yield 'None' to signal the end of the generator and stop iteration.
    # This helps to avoid raising StopIteration errors when the generator is exhausted.
    yield None

def get_frame(filename, index):
    # Initialize a counter to keep track of the frames processed.
    counter = 0

    # Create a VideoCapture object 'video' to open the video file for reading.
    video = cv2.VideoCapture(filename)

    # Use a while loop to continuously read frames from the video until it's open.
    while video.isOpened():
        # Read the next video frame.
        ret, frame = video.read()

        # If 'ret' is True, the frame was read successfully.
        if ret:
            # Check if the current frame's index matches the desired index.
            if counter == index:
                # Return the frame if the index matches.
                return frame

            # Increment the counter to move to the next frame.
            counter += 1
        else:
            # If 'ret' is False, it means there are no more frames to read in the video,
            # so we break the loop and return None to indicate that the desired index is out of range.
            break

    # Release the VideoCapture object to free up resources.
    video.release()

    # If the desired index is out of range, return None.
    return None

if __name__ == "__main__":

    initFrame = get_frame(VFILE, 1)
    initL1 = []
    initL2 = []
    for ligne in initFrame[:,400:850]:
        initL1.append(list(ligne[200]))
        initL2.append(list(ligne[300]))
    initL1 = np.array(initL1)
    initL2 = np.array(initL2)
    #print(np.array(initL1))
    #print(signal.correlate2d(initL1,initL1))
    #print(np.corrcoef(initL1, initL1))




    # Loop through each frame in the video using the generator 'get_frames(VFILE)'.
    for f in get_frames(VFILE):
        # If 'f' is None, it means there are no more frames to process, so we break the loop.
        if f is None:
            break
        # Display the current frame using OpenCV's 'imshow' function.
        # The first argument is the window name, which will be displayed at the top of the window.
        # The second argument is the frame to be displayed.

        f = f[:,400:850]
        f1 = f.copy()
        L1 = []
        L2 = []
        for ligne in f1:
            L1.append(list(ligne[200]))
            L2.append(list(ligne[300]))
            ligne[200] = [250,250,250]

        L1 = np.array(L1)
        L2 = np.array(L2)
        print("L1",np.allclose(L1, initL1,rtol=0, atol=50))
        print("L2",np.allclose(L2, initL2,rtol=0, atol=50))
        cv2.imshow('frame', f1)

        # Wait for a short time (10 milliseconds) and check for the 'Esc' key press (ASCII code 27).
        # If the 'Esc' key is pressed, we break the loop to stop displaying frames.
        if cv2.waitKey(10) == 27:
            break

    # After the loop ends, close all OpenCV windows using 'destroyAllWindows'.
    # This is necessary to release any resources held by the windows.
    cv2.destroyAllWindows()

    # Using the 'get_frame' function to retrieve a specific frame from the video file.
    # We pass the video file path 'VFILE' and the desired frame index '80' as arguments.
    frame = get_frame(VFILE, 80)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Printing the shape of the 'frame' array, which represents the dimensions of the frame.
    # The shape will be in the format (height, width, channels).
    # 'height' represents the number of rows (vertical pixels), 'width' represents the number of columns (horizontal pixels),
    # and 'channels' represents the number of color channels (e.g., 3 for RGB).
    print('shape ', frame.shape)

    # Printing the pixel value at position (0,0) in the frame.
    # The format of the pixel value will depend on the number of color channels.
    # For example, if the image is in RGB format, the pixel value will be an array containing 3 values (R, G, B).
    print('pixel at (0,0)', frame[600, 600, :])
    plt.imshow(frame)
    plt.show()