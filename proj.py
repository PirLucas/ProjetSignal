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
    """
    Generator function to read frames from a video file.

    :param filename: Path to the video file.
    :yield: Yields each frame of the video as a numpy array.
    """
    video = None
    try:
        # Create a VideoCapture object to open the video file for reading.
        video = cv2.VideoCapture(filename)

        # Continuously read frames from the video until it's open.
        while video.isOpened():
            # Read the next video frame.
            ret, frame = video.read()

            # If the frame was read successfully, yield it.
            if not ret:
                break
            yield frame
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Release the VideoCapture object to free up resources.
        if video:
            video.release()


def get_frame(filename, index):
    """
    Retrieves a specific frame from a video file.

    :param filename: Path to the video file.
    :param index: The index of the frame to retrieve.
    :return: The specified frame as a numpy array, or None if the index is out of range.
    """
    video = None
    try:
        # Create a VideoCapture object to open the video file for reading.
        video = cv2.VideoCapture(filename)

        # Set the position of the next frame to be read.
        video.set(cv2.CAP_PROP_POS_FRAMES, index)

        # Read the frame at the specified index.
        ret, frame = video.read()

        # Return the frame if it was read successfully.
        if ret:
            return frame
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Release the VideoCapture object to free up resources.
        if video:
            video.release()

    # Return None if the frame could not be retrieved.
    return None

if __name__ == "__main__":
    # Retrieve the first frame, second frame, and the 292nd frame from the video.
    initFrame = get_frame(VFILE, 1)
    initFrame2 = get_frame(VFILE, 292)
    initFrame3 = get_frame(VFILE, 2)

    # Convert these frames to grayscale.
    grayFrame = cv2.cvtColor(initFrame, cv2.COLOR_BGR2GRAY)
    initFrame2 = cv2.cvtColor(initFrame2, cv2.COLOR_BGR2GRAY)
    initFrame3 = cv2.cvtColor(initFrame3, cv2.COLOR_BGR2GRAY)

    # Extract specific pixel values from grayFrame for comparison.
    initL1 = []
    initL2 = []
    for ligne in grayFrame[:, 400:850]:
        initL1.append(ligne[200])
        initL2.append(ligne[300])
    initL1 = np.array(initL1)
    initL2 = np.array(initL2)

    # Calculate and print the correlation coefficients.
    print(np.corrcoef(grayFrame, grayFrame))
    print(np.corrcoef(grayFrame, initFrame2))

    # Compute the difference in correlation coefficients between frames.
    np_subtr = np.subtract(np.corrcoef(grayFrame, grayFrame), np.corrcoef(grayFrame, initFrame3))
    np_mean = np.mean(np_subtr)
    print('numpy mean is:', np_mean)

    # Display the difference in correlation coefficients.
    plt.imshow(np_subtr)
    plt.show()

    # Loop through each frame in the video.
    for e, f in enumerate(get_frames(VFILE)):
        if f is None:
            break

        # Process the frame and extract specific pixel values.
        f = f[:, 400:850]
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        f1 = f.copy()
        L1 = []
        L2 = []

        for ligne in f1:
            L1.append(ligne[200])
            L2.append(ligne[300])
            # Highlight specific pixel values in the frame.
            ligne[200] = 250
            ligne[300] = 250

        L1 = np.array(L1)
        L2 = np.array(L2)

        # Calculate the difference in correlation coefficients and display it.
        diff = np.subtract(np.corrcoef(grayFrame[:, 400:850], grayFrame[:, 400:850]), np.corrcoef(grayFrame[:, 400:850], f))
        diff2 = np.corrcoef(grayFrame[:, 400:850], f)
        res = cv2.resize(diff, dsize=(450, 450), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('frame', res)
        cv2.imshow('frame2', f1)

        if cv2.waitKey(10) == 27:  # Check for 'Esc' key press to exit the loop.
            break

    cv2.destroyAllWindows()

    # Retrieve and display a specific frame (index 80) from the video.
    frame_80 = get_frame(VFILE, 80)
    frame_80_rgb = cv2.cvtColor(frame_80, cv2.COLOR_BGR2RGB)
    print('shape', frame_80_rgb.shape)
    print('pixel at (600,600)', frame_80_rgb[600, 600, :])
    plt.imshow(frame_80_rgb)
    plt.show()