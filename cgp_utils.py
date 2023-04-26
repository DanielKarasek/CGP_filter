import pickle
from typing import Tuple

import numpy as np
from cgp.individual import IndividualBase

from image_setup_utils import create_sliding_window


def save_cgp(indiv: IndividualBase, filename: str):
  with open(filename, 'wb') as f:
    pickle.dump(indiv, f)


def load_cgp(filename: str) -> IndividualBase:
  with open(filename, 'rb') as f:
    return pickle.load(f)


def detect_noise(cgp: IndividualBase, sliding_window: np.ndarray, accept_probability: float = 0.5) -> np.ndarray:
  f = cgp.to_numpy()
  raw_estimates = f(sliding_window.T)
  probability_estimates = 1/(1+np.exp(-raw_estimates))
  return np.where(probability_estimates > accept_probability, 1, 0)


def detect_from_saved(cgp_file_path: str, noised_image: np.ndarray, sliding_window_size: int) -> np.ndarray:
  cgp = load_cgp(cgp_file_path)
  sliding_window = create_sliding_window(noised_image, sliding_window_size)
  detected_binary_vector = detect_noise(cgp, sliding_window)
  return detected_binary_vector


def detect_indices_from_saved(cgp_file_path: str, noised_image: np.ndarray, sliding_window_size: int) -> np.ndarray:
  cgp = load_cgp(cgp_file_path)
  sliding_window = create_sliding_window(noised_image, sliding_window_size)
  detected_binary_vector = detect_noise(cgp, sliding_window)
  return detected_binary_vector.reshape(noised_image)


def filter_noise(cgp: IndividualBase,
                 sliding_window: np.ndarray) -> np.ndarray:
  f = cgp.to_numpy()
  raw_estimates = f(sliding_window.T)
  filtered_values = np.clip(raw_estimates, 0, 1)
  filtered_values = filtered_values.astype(float)
  return filtered_values


def filter_noise_from_saved(detect_cgp_filepath: str,
                            filter_cgp_filepath: str,
                            noised_image: np.ndarray,
                            sliding_window_size: int,
                            accept_probability: float) -> Tuple[np.ndarray, np.ndarray]:
  detect_cgp = load_cgp(detect_cgp_filepath)
  sliding_window = create_sliding_window(noised_image, sliding_window_size)
  sliding_window = np.reshape(sliding_window, (sliding_window.shape[0], -1))
  detected_binary_vector = detect_noise(detect_cgp, sliding_window, accept_probability)
  sliding_window = sliding_window[detected_binary_vector.astype(bool)]
  filter_cgp = load_cgp(filter_cgp_filepath)
  filtered_values_vector = filter_noise(filter_cgp, sliding_window)

  return detected_binary_vector, filtered_values_vector


def restore_from_saved(detect_cgp_filepath: str,
                        filter_cgp_filepath: str,
                        noised_image: np.ndarray,
                        sliding_window_size: int,
                        accept_probability: float) -> np.ndarray:
  detected_binary_vector, filtered_values_vector = filter_noise_from_saved(detect_cgp_filepath,
                                                                           filter_cgp_filepath,
                                                                           noised_image,
                                                                           sliding_window_size,
                                                                           accept_probability)
  restored_image = restore_image(detected_binary_vector, filtered_values_vector, noised_image)
  return restored_image

def restore_image(detected_binary_vector: np.ndarray,
                  filtered_values_vector: np.ndarray,
                  noised_image: np.ndarray) -> np.ndarray:
  restored_image = noised_image.copy()
  restored_image = restored_image.reshape((-1))
  restored_image[detected_binary_vector.astype(bool)] = filtered_values_vector
  restored_image = restored_image.reshape(noised_image.shape)
  return restored_image
