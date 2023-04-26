from typing import Optional, Tuple

import cv2
import numpy as np

import image_noising


def downscale_image(image: np.ndarray, scale_percent: float) -> np.ndarray:
    width = int(image.shape[1] * scale_percent)
    height = int(image.shape[0] * scale_percent)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def create_dataset_from_image(path_to_image: str, window_size) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  np.random.seed(35784)
  original_image = cv2.imread(path_to_image)[:, :, 0]
  original_image.squeeze()
  original_downscaled_image = downscale_image(original_image, 0.3)
  noised_downscaled_image = image_noising.hurl_noise(0.2, np.copy(original_downscaled_image))
  noised_sliding_window = create_sliding_window(noised_downscaled_image, window_size)

  original_flatenned = original_downscaled_image.reshape(-1)

  return original_flatenned, noised_sliding_window, noised_downscaled_image


def detection_image_processing(target_image: np.ndarray) -> np.ndarray:
  target_image = np.asarray(target_image, float)
  target_image /= 255
  return target_image


def create_detection_dataset_from_image(path_to_image: str, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
  original_flatenned, noised_sliding_window, _ = create_dataset_from_image(path_to_image, window_size)
  original_flatenned = detection_image_processing(original_flatenned)
  noised_sliding_window = detection_image_processing(noised_sliding_window)
  return noised_sliding_window, original_flatenned


def create_sliding_window(image: np.ndarray, window_size: int) -> np.ndarray:
  pad_size = window_size//2
  padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), 'reflect')
  image_sliding_win = (
    np.lib.stride_tricks.sliding_window_view(padded_image, (window_size, window_size)))
  return image_sliding_win.reshape((-1, window_size, window_size))


def regression_processing(target_image: np.ndarray) -> np.ndarray:
  target_image = np.asarray(target_image, float)
  target_image /= 255
  # target_image -= 0.5
  return target_image


def create_regression_dataset(path_to_image: str, window_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  dataset_y, dataset_x, noised_downscaled = create_dataset_from_image(path_to_image, window_size)
  filter_vector = np.where(dataset_x[:, window_size//2, window_size//2] == dataset_y, 0, 1).astype(bool)

  dataset_x = dataset_x[filter_vector]
  dataset_y = dataset_y[filter_vector]

  noised_downscaled = regression_processing(noised_downscaled)
  dataset_x = regression_processing(dataset_x)
  dataset_y = regression_processing(dataset_y)
  return dataset_x, dataset_y, noised_downscaled, filter_vector


if __name__ == "__main__":
  dataset_x, dataset_y = create_dataset_from_image('lenna.png')
