import pickle

import numpy as np
from cgp.individual import IndividualBase

from image_setup_utils import create_sliding_window


def save_cgp(individual: IndividualBase, filename: str):
  """
  Saves CGP to file
  :param individual:
  :param filename:
  :return:
  """
  with open(filename, 'wb') as f:
    pickle.dump(individual, f)


def load_cgp(filename: str) -> IndividualBase:
  """
  Loads CGP from file
  :param filename:
  :return: Saved individual
  """
  with open(filename, 'rb') as f:
    return pickle.load(f)


def detect_noise(cgp: IndividualBase,
                 sliding_window: np.ndarray,
                 accept_probability: float = 0.5) -> np.ndarray:
  """
  Detects noise in image by CGP
  :param cgp:
  :param sliding_window: np.ndarray of shape (N, M) where M is the size of sliding window (e.g. 3x3 = 9)
                         and N is the number of sliding windows
  :param accept_probability: threhold probability for accepting the noise
  :return: binary vector of shape (N,) where N is the number of sliding windows representing
           positions of noise
  """

  f = cgp.to_numpy()
  raw_estimates = f(sliding_window.T)
  probability_estimates = 1/(1+np.exp(-raw_estimates))
  return np.where(probability_estimates > accept_probability, 1, 0)


def detect_from_saved(cgp_file_path: str, noised_image: np.ndarray, sliding_window_size: int) -> np.ndarray:
  """
  Detects noise in image by saved CGP
  :param cgp_file_path:
  :param noised_image: Grey scale image of shape (H, W) where H is the number of rows and W is the number of columns
  :param sliding_window_size: np.ndarray of shape (N, M) where M is the size of sliding window (e.g. 3x3 = 9)
                         and N is the number of sliding windows
  :return: binary vector of shape (N,) where N is the number of sliding windows representing
           positions of noise
  """
  cgp = load_cgp(cgp_file_path)
  sliding_window = create_sliding_window(noised_image, sliding_window_size)
  detected_binary_vector = detect_noise(cgp, sliding_window)
  return detected_binary_vector


def detection_mask_from_saved(cgp_file_path: str, noised_image: np.ndarray, sliding_window_size: int) -> np.ndarray:
  """
  Detects noise in image by saved CGP and returns it as a mask with shape of the noisy image
  :param cgp_file_path:
  :param noised_image: Grey scale image of shape (H, W) where H is the number of rows and W is the number of columns
  :param sliding_window_size: np.ndarray of shape (N, M) where M is the size of sliding window (e.g. 3x3 = 9)
                          and N is the number of sliding windows
  :return: binary mask of shape (H, W) where H is the number of rows and W is the number of columns in the noisy image
  """
  return detect_from_saved(cgp_file_path, noised_image, sliding_window_size).reshape(noised_image)


def filter_noise(cgp: IndividualBase,
                 sliding_window: np.ndarray) -> np.ndarray:
  """
  Filters noise from sliding window
  :param cgp:
  :param sliding_window: np.ndarray of shape (N, M) where M is the size of sliding window (e.g. 3x3 = 9)
                          and N is the number of sliding window
  :return: np.ndarray of shape (N,) where N is the number of sliding windows representing filtered values
  """
  f = cgp.to_numpy()
  raw_estimates = f(sliding_window.T)
  filtered_values = np.clip(raw_estimates, 0, 1)
  filtered_values = filtered_values.astype(float)
  return filtered_values


def restore_from_saved(detect_cgp_filepath: str,
                       filter_cgp_filepath: str,
                       noised_image: np.ndarray,
                       sliding_window_size: int,
                       accept_probability: float) -> np.ndarray:
  """
  Restores image using saved detect and filter CGPs
  :param detect_cgp_filepath:
  :param filter_cgp_filepath:
  :param noised_image: Grey scale image with shape (H, W), where H represents height and W represents width
  :param sliding_window_size: Size of sliding window e.g. (3 ==> 3x3 = 9)
  :param accept_probability: Threshold probability for accepting the noise (lower --> more sensitive detector
                             and vice versa)
  :return: restored input image
  """
  detect_cgp = load_cgp(detect_cgp_filepath)
  sliding_window = create_sliding_window(noised_image, sliding_window_size)
  sliding_window = np.reshape(sliding_window, (sliding_window.shape[0], -1))
  detected_binary_vector = detect_noise(detect_cgp, sliding_window, accept_probability)
  sliding_window = sliding_window[detected_binary_vector.astype(bool)]
  filter_cgp = load_cgp(filter_cgp_filepath)
  filtered_values_vector = filter_noise(filter_cgp, sliding_window)
  restored_image = restore_image(detected_binary_vector, filtered_values_vector, noised_image)
  return restored_image


def restore_image(detected_binary_vector: np.ndarray,
                  filtered_values_vector: np.ndarray,
                  noised_image: np.ndarray) -> np.ndarray:
  """
  Restores image using detected binary vector and filtered values vector
  :param detected_binary_vector: binary vector of shape (N,), which represents detected noise pixels
  :param filtered_values_vector: values that should be used to replace the noise pixels
  :param noised_image:
  :return: restored input image
  """
  restored_image = noised_image.copy()
  restored_image = restored_image.reshape((-1))
  restored_image[detected_binary_vector.astype(bool)] = filtered_values_vector
  restored_image = restored_image.reshape(noised_image.shape)
  return restored_image
