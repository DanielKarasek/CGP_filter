import cgp
from cgp.individual import IndividualBase
import pickle

import numpy as np
from image_setup_utils import create_sliding_window


def save_cgp(indiv: IndividualBase, filename: str):
  with open(filename, 'wb') as f:
    pickle.dump(indiv, f)


def load_cgp(filename: str) -> IndividualBase:
  with open(filename, 'rb') as f:
    return pickle.load(f)


def detect_noise(cgp: IndividualBase, sliding_window: np.ndarray) -> np.ndarray:
  f = cgp.to_func()
  raw_estimates = np.array([f(*x) for x in sliding_window])
  probability_estimates = 1/(1+np.exp(-raw_estimates))
  return np.where(probability_estimates > 0.5, 1, 0)


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
  filtered_values = np.clip(raw_estimates, -0.5, 0.5)
  filtered_values = filtered_values.astype(float)
  return filtered_values


def filter_noise_from_saved(detect_cgp_filepath: str,
                            filter_cgp_filepath: str,
                            noised_image: np.ndarray,
                            sliding_window_size: int) -> np.ndarray:
  detect_cgp = load_cgp(detect_cgp_filepath)
  sliding_window = create_sliding_window(noised_image, sliding_window_size)
  detected_binary_vector = detect_noise(detect_cgp, sliding_window)
  sliding_window = sliding_window[detected_binary_vector.astype(bool)]
  filter_cgp = load_cgp(filter_cgp_filepath)
  filtered_values_vector = filter_noise(filter_cgp, sliding_window)

  return filtered_values_vector


def restore_image(detected_binary_vector: np.ndarray,
                  filtered_values_vector: np.ndarray,
                  noised_image: np.ndarray) -> np.ndarray:
  restored_image = noised_image.copy()
  restored_image = restored_image.reshape((-1))
  restored_image[detected_binary_vector.astype(bool)] = filtered_values_vector
  restored_image = restored_image.reshape(noised_image.shape)
  return restored_image
