import copy
import multiprocessing
import threading

import cv2
import numpy as np
import scipy.signal
from scipy.signal import convolve2d

class PostDetector:
    __frames: list[np.ndarray]
    __logo_frame: np.ndarray

    def __init__(self, frames, logo_frame):
        self.__frames = logo_frame
        self.__logo_frame = np.empty_like(logo_frame)
        np.copyto(self.__logo_frame, logo_frame)


class LogoDetector:
    __frames: list[np.ndarray] = []
    __logo_frame: np.ndarray
    __correlation_threshold: int
    __is_in_calcul: bool = False

    def __init__(self, correlation_acceptable_value, logo_image_path: str):
        self.__correlation_threshold = correlation_acceptable_value
        logo = cv2.imread(logo_image_path)
        self.__logo_frame = np.fft.fft2(
            np.rot90(
                logo,
                2
            )
        )

    def add_frame(self, frame: np.ndarray) -> None:
        self.__frames.append(copy.copy(frame))

    def calcul_convolution_async(self):
        self.__is_in_calcul = True
        thread = threading.Thread(target=lambda: self.calcul_convolutions())
        thread.start()

    def calcul_convolutions(self):
        detected: int = 0
        print("Longueurs : ", len(self.__frames))

        for frame in self.__frames:
            cv2.destroyAllWindows()
            cv2.imshow("Logo trouvé", frame)
            correlation = self.get_correlation(frame, self.__logo_frame)
            print(f"{correlation} : {'Logo found' if correlation >= self.__correlation_threshold else 'Logo not found'}")
            if correlation >= self.__correlation_threshold:
                detected += 1
        return detected

    @staticmethod
    def get_correlation(frame: np.ndarray, logo: np.ndarray) -> float:

        return np.ndarray.max(
            np.real(
                np.fft.ifft2(
                    convolve2d(
                        np.fft.fft2(frame),
                        logo,
                    )
                )
            )
        )



    @property
    def frames(self):
        return self.__frames

    def can_capture_frames(self) -> bool:
        return len(self.__frames) < self.__max_total_frames and not self.__is_in_calcul

