import copy
import threading

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from LogoDetector2 import LogoDetector
import time
from skimage import filters, color, measure, exposure

VFILE = "projet/logo.mp4"


def get_frame_ratio(frame: np.ndarray) -> float:
    filled = ndimage.binary_fill_holes(get_binary_frame(frame), structure=np.ones((3, 3)))
    label, n = measure.label(filled, return_num=True)
    props = measure.regionprops(label)
    region = max(props, key=lambda p: p.area)
    return region.axis_major_length / region.axis_minor_length


def get_frame_ratios(frame: np.ndarray) -> list[float]:
    filled = ndimage.binary_fill_holes(get_binary_frame(frame), structure=np.ones((3, 3)))
    label, n = measure.label(filled, return_num=True)
    props = measure.regionprops(label)
    return [
        region.axis_major_length / region.axis_minor_length
        if region.axis_major_length != 0 and region.axis_minor_length != 0
        else 0
        for region in props
    ]


def check_ratio(ratios: list[float], good_ratio: float, tolerance: float) -> bool:
    for ratio in ratios:
        if good_ratio - tolerance <= ratio <= good_ratio + tolerance:
            return True
    return False


def check_frame(frame: np.ndarray, ratio) -> bool:
    return check_ratio(get_frame_ratios(frame), ratio, 0.0345)

def get_binary_frame(frame: np.ndarray) -> np.ndarray:
    gray = color.rgb2gray(frame)
    return gray > filters.threshold_otsu(gray)


def get_luminosity(frame: np.ndarray) -> tuple:
    rgb = [0, 0, 0]
    for i in range(len(frame)):
        for j in frame[i]:
            rgb[0] += j[0]
            rgb[1] += j[1]
            rgb[2] += j[2]
    length = len(frame) * len(frame[0])
    return rgb[0] / length, rgb[1] / length, rgb[2] / length


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
    corr_base = np.corrcoef(base_line, base_line)
    corr_current = np.corrcoef(base_line, current_line)
    return np.subtract(corr_base, corr_current)


def process_and_display_frames(video_file, base_frame):
    enter_count = 0
    exit_count = 0
    line1_crossed, line2_crossed = False, False
    frame_counter, debounce_frames = 0, 60  # debounce frames, number of frames to ignore after a crossing is detected
    line1_position, line2_position = 270, 100  # Adjust as needed
    correlation_threshold = 0.20  # Adjust based on your testing
    correlations = []  # List to store correlation values
    first_line_crossed = None  # Keeps track of which line was crossed first
    logo_ratio: float = get_frame_ratio(cv2.imread('projet/logo4.jpg'))

    previous_check: bool = False

    for frame in get_frames(video_file):
        has_logo: bool | None = None
        if frame is None:
            break

        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 400:850

        # Calculate correlation changes at the line positions
        corr_change_line1 = calculate_correlation_change(processed_frame, base_frame, line1_position)
        corr_change_line2 = calculate_correlation_change(processed_frame, base_frame, line2_position)

        # Store the mean of the correlation changes
        correlations.append(np.mean([corr_change_line1, corr_change_line2]))
        numpy_mean1 = np.mean(corr_change_line1)
        numpy_mean2 = np.mean(corr_change_line2)

        print(corr_change_line1, numpy_mean1, corr_change_line2, numpy_mean2)

        # Detect line crossing with debounce
        if numpy_mean1 > correlation_threshold and not line1_crossed and frame_counter > debounce_frames:
            line1_crossed = True
            if not line2_crossed:  # Only start crossing process if line2 hasn't been crossed yet
                first_line_crossed = 1
            frame_counter = 0  # Reset frame counter

        if numpy_mean2 > correlation_threshold and not line2_crossed and frame_counter > debounce_frames:
            line2_crossed = True
            if not line1_crossed:  # Only start crossing process if line1 hasn't been crossed yet
                first_line_crossed = 2
            frame_counter = 0  # Reset frame counter

        # Check sequence for entering or exiting
        if first_line_crossed:
            if first_line_crossed == 1 and line2_crossed:
                exit_count += 1  # Exiting
                has_logo = check_frame(frame, logo_ratio)
                previous_check = has_logo
                print("Person Exited")
                line1_crossed, line2_crossed, first_line_crossed = False, False, None

            elif first_line_crossed == 2 and line1_crossed:
                enter_count += 1  # Entering
                has_logo = check_frame(frame, logo_ratio)
                previous_check = has_logo
                print("Person Entered")
                line1_crossed, line2_crossed, first_line_crossed = False, False, None

        frame_counter += 1

        # Display frames with lines and counts
        cv2.line(processed_frame, (line1_position, 0), (line1_position, processed_frame.shape[0]), (255, 0, 0), 2)
        cv2.line(processed_frame, (line2_position, 0), (line2_position, processed_frame.shape[0]), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(processed_frame, f'Entering: {enter_count}', (10, 30), font, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(processed_frame, f'Exiting: {exit_count}', (10, 70), font, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(processed_frame, f'Mean1: {numpy_mean1:.2f}', (10, 110), font, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(processed_frame, f'Mean2: {numpy_mean2:.2f}', (10, 150), font, 1, (255, 0, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(processed_frame, f"Logo ? : {has_logo if has_logo is not None else previous_check}", (10, 230), font, 1, (255, 0, 0),
                    2, cv2.LINE_AA)
        cv2.imshow("Image", processed_frame)

        frame_counter += 1

        if cv2.waitKey(int(2)) == 27:  # ESC key
            break

    cv2.destroyAllWindows()
    # print(f"{detector.calcul_convolutions()} logos detected")
    return enter_count, exit_count, correlations


if __name__ == "__main__":
    gray_frame = getGrayscaleFrame(VFILE, 1)

    enter_count, exit_count, correlations = process_and_display_frames(VFILE, gray_frame)
    print(f"People Entered: {enter_count}, People Exited: {exit_count}", f"max correlation: {max(correlations)}")
