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


def getGrayscaleFrame(video_file, frame_index):
    """Retrieve a specific frame from a video file and convert it to grayscale."""
    frame = get_frame(video_file, frame_index)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame is not None else None


def display_correlation_difference(baseFrame, compareFrame, title):
    """Calculate and display the correlation difference between two frames."""
    correlation_diff = np.subtract(np.corrcoef(baseFrame, baseFrame), 
                                   np.corrcoef(baseFrame, compareFrame))
    plt.imshow(correlation_diff)
    plt.title(title)
    plt.show()


def calculate_correlation_change(current_frame, base_frame, line_position):
    """
    Calculate the correlation change at a specific line position.
    :param current_frame: The current frame to compare.
    :param base_frame: The base frame for comparison.
    :param line_position: The position of the line to check.
    :return: The correlation change at the line position.
    """
    base_line = base_frame[:, line_position]
    current_line = current_frame[:, line_position]
    corr_base = np.corrcoef(base_line, base_line)[0, 1]
    corr_current = np.corrcoef(base_line, current_line)[0, 1]
    return np.subtract(corr_base, corr_current)


def process_and_display_frames(video_file, base_frame):
    enter_count = 0
    exit_count = 0
    line1_crossed, line2_crossed = False, False
    frame_counter, debounce_frames = 0, 30 # debounce frames, number of frames to ignore after a crossing is detected
    line1_position, line2_position = 135, 420  # Adjust as needed
    correlation_threshold = 0.96  # Adjust based on your testing
    correlations = []  # List to store correlation values
    first_line_crossed = None  # Keeps track of which line was crossed first


    for frame in get_frames(video_file):
        if frame is None:
            break

        processed_frame = cv2.cvtColor(frame[:, 400:850], cv2.COLOR_BGR2GRAY)

        # Calculate correlation changes at the line positions
        corr_change_line1 = calculate_correlation_change(processed_frame, base_frame[:, 400:850], line1_position)
        corr_change_line2 = calculate_correlation_change(processed_frame, base_frame[:, 400:850], line2_position)

        # Store the mean of the correlation changes
        correlations.append(np.mean([corr_change_line1, corr_change_line2]))
        numpy_mean1 = np.mean(corr_change_line1)
        numpy_mean2 = np.mean(corr_change_line2)


        # Detect line crossing with debounce
        if corr_change_line1 > correlation_threshold and not line1_crossed and frame_counter > debounce_frames:
            line1_crossed = True
            if not line2_crossed:  # Only start crossing process if line2 hasn't been crossed yet
                first_line_crossed = 1
            frame_counter = 0  # Reset frame counter

        if corr_change_line2 > correlation_threshold and not line2_crossed and frame_counter > debounce_frames:
            line2_crossed = True
            if not line1_crossed:  # Only start crossing process if line1 hasn't been crossed yet
                first_line_crossed = 2
            frame_counter = 0  # Reset frame counter

        # Check sequence for entering or exiting
        if first_line_crossed:
            if first_line_crossed == 1 and line2_crossed:
                exit_count += 1  # Exiting
                print("Person Exited")
                line1_crossed, line2_crossed, first_line_crossed = False, False, None

            elif first_line_crossed == 2 and line1_crossed:
                enter_count += 1  # Entering
                print("Person Entered")
                line1_crossed, line2_crossed, first_line_crossed = False, False, None

        frame_counter += 1

        # Display frames with lines and counts
        cv2.line(processed_frame, (line1_position, 0), (line1_position, processed_frame.shape[0]), (255, 0, 0), 2)
        cv2.line(processed_frame, (line2_position, 0), (line2_position, processed_frame.shape[0]), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(processed_frame, f'Entering: {enter_count}', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(processed_frame, f'Exiting: {exit_count}', (10, 70), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(processed_frame, f'Mean1: {numpy_mean1:.2f}', (10, 110), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(processed_frame, f'Mean2: {numpy_mean2:.2f}', (10, 130), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Processed Frame', processed_frame)

        frame_counter += 1

        if cv2.waitKey(int(33 * 2)) == 27:  # ESC key
            break

    cv2.destroyAllWindows()
    return enter_count, exit_count, correlations

if __name__ == "__main__":
    gray_frame = getGrayscaleFrame(VFILE, 1)

    enter_count, exit_count, correlations = process_and_display_frames(VFILE, gray_frame)
    print(f"People Entered: {enter_count}, People Exited: {exit_count}", f"max correlation: {max(correlations)}")