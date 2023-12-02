import copy
import multiprocessing
import threading
import time

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

    def __init__(self, correlation_acceptable_value, logo_image_path: str):
        self.__correlation_threshold = correlation_acceptable_value
        self.__logo_frame = np.fft.fft2(
            np.rot90(
                self.apply_filter(cv2.imread(logo_image_path, cv2.IMREAD_UNCHANGED), (168, 230, 243), tolerance=80),
                2
            )
        )

    def add_frame(self, frame: np.ndarray) -> None:
        self.__frames.append(copy.copy(frame))

    def calcul_convolutions(self):
        detected: int = 0
        print("Longueurs : ", len(self.__frames))

        for frame in self.__frames:
            cv2.destroyAllWindows()
            correlation = self.get_correlation(frame, self.__logo_frame)
            print(f"{correlation} : {'Logo found' if correlation >= self.__correlation_threshold else 'Logo not found'}")
            if correlation >= self.__correlation_threshold:
                detected += 1
        return detected

    @staticmethod
    def get_correlation(frame: np.ndarray, logo: np.ndarray) -> float:
        copy = LogoDetector.apply_filter(frame, (168, 230, 243), tolerance=80)
        return np.ndarray.max(
            np.real(
                np.fft.ifft2(
                    convolve2d(
                        np.fft.fft2(copy),
                        logo,
                    )
                )
            )
        )

    @staticmethod
    def apply_filter(frame: np.ndarray, rgb: tuple, tolerance=10) -> np.ndarray:
        tresh = cv2.inRange(
            frame,
            np.array([rgb[0] - tolerance, rgb[1] - tolerance, rgb[2] - tolerance]),
            np.array([rgb[1] + tolerance, rgb[2] + tolerance, rgb[2] + tolerance])
        )
        return 255 - tresh
