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


def get_grayscale_frame(video_file, frame_index):
    """Retrieve a specific frame from a video file and convert it to grayscale."""
    frame = get_frame(video_file, frame_index)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame is not None else None


def display_correlation_difference(base_frame, compare_frame, title):
    """Calculate and display the correlation difference between two frames."""
    correlation_diff = np.subtract(np.corrcoef(base_frame, base_frame), 
                                   np.corrcoef(base_frame, compare_frame))
    plt.imshow(correlation_diff)
    plt.title(title)
    plt.show()


def process_and_display_frames(video_file, base_frame):
    """Process each frame in the video and display the correlation differences."""
    window_name_diff = 'Frame Difference'
    window_name_processed = 'Processed Frame'

    for e, frame in enumerate(get_frames(video_file)):
        if frame is None:
            break

        processed_frame = cv2.cvtColor(frame[:, 400:850], cv2.COLOR_BGR2GRAY)
        processed_frame[:, [200, 300]] = 250  # Highlight specific pixel values.

        # Calculate the correlation differences and display in the same window.
        diff = np.subtract(np.corrcoef(base_frame[:, 400:850], base_frame[:, 400:850]),
                           np.corrcoef(base_frame[:, 400:850], processed_frame))
        cv2.imshow(window_name_diff, cv2.resize(diff, dsize=(450, 450), interpolation=cv2.INTER_CUBIC))
        cv2.imshow(window_name_processed, processed_frame)

        if cv2.waitKey(10) == 27:  # ESC key
            break

    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    grayFrame = get_grayscale_frame(VFILE, 1)
    initFrame2 = get_grayscale_frame(VFILE, 292)
    initFrame3 = get_grayscale_frame(VFILE, 2)

    np_subtr = np.subtract(np.corrcoef(grayFrame, grayFrame), np.corrcoef(grayFrame, initFrame3))
    print('numpy mean is:', np.mean(np_subtr))
    display_correlation_difference(grayFrame, initFrame3, 'Correlation Difference')

    process_and_display_frames(VFILE, grayFrame)

    frame_80_rgb = cv2.cvtColor(get_grayscale_frame(VFILE, 80), cv2.COLOR_BGR2RGB)
    print('shape', frame_80_rgb.shape)
    print('pixel at (600,600)', frame_80_rgb[600, 600, :])
    plt.imshow(frame_80_rgb)
    plt.show()