from typing import Tuple

import cv2
import numpy as np

import image_noising


def downscale_image(image: np.ndarray, scale_percent: float) -> np.ndarray:
    """
    Downscales image by scale_percent
    :param image:
    :param scale_percent:
    :return:
    """
    width = int(image.shape[1] * scale_percent)
    height = int(image.shape[0] * scale_percent)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def create_dataset_from_image(path_to_image: str, window_size) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
  Creates dataset from image -> this consisst of downscaling, noising and creating sliding window.
  :param path_to_image:
  :param window_size: size of sliding window
  :return: Downscaled image flattened, noised sliding window, noised downscaled image
  """
  np.random.seed(35784)
  original_image = cv2.imread(path_to_image)[:, :, 0]
  original_image.squeeze()
  original_downscaled_image = downscale_image(original_image, 1)
  noised_downscaled_image = image_noising.hurl_noise(0.2, np.copy(original_downscaled_image))
  noised_sliding_window = create_sliding_window(noised_downscaled_image, window_size)

  original_flatenned = original_downscaled_image.reshape(-1)

  return original_flatenned, noised_sliding_window, noised_downscaled_image


def detection_image_processing(target_image: np.ndarray) -> np.ndarray:
  """
  Processes image before we try to detect noisy in it.
  :param target_image:
  :return:
  """
  target_image = np.asarray(target_image, float)
  target_image /= 255
  return target_image


def create_detection_dataset_from_image(path_to_image: str, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
  """
  Creates dataset from image -> First of all dataset is created and then it's processed for detection
  :param path_to_image:
  :param window_size:
  :return:
  """
  original_flatenned, noised_sliding_window, _ = create_dataset_from_image(path_to_image, window_size)
  original_flatenned = detection_image_processing(original_flatenned)
  noised_sliding_window = detection_image_processing(noised_sliding_window)
  return noised_sliding_window, original_flatenned


def create_sliding_window(image: np.ndarray, window_size: int) -> np.ndarray:
  """
  Creates sliding window with padding from image
  :param image:
  :param window_size: size of the sliding window
  :return:
  """
  pad_size = window_size//2
  padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), 'reflect')
  image_sliding_win = (
    np.lib.stride_tricks.sliding_window_view(padded_image, (window_size, window_size)))
  return image_sliding_win.reshape((-1, window_size, window_size))


def regression_processing(target_image: np.ndarray) -> np.ndarray:
  """
  Processes image before we try to repair the noise in it.
  :param target_image:
  :return:
  """
  target_image = np.asarray(target_image, float)
  target_image /= 255
  return target_image


def create_regression_dataset(path_to_image: str, window_size: int) -> (
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
  """
  Creates dataset and then process it for regression
  :param path_to_image:
  :param window_size: size of the sliding window
  :return: dataset_x (sliding window shape (n, window_size, window_size)), dataset_y correct valules for dataset_x,
  noised_downscaled image, filter_vector (binary vector which tells what positions int noised_downscaled image are
  noisy == which positions we want to repair == dataset_x and dataset_y positions in original image)
  """
  dataset_y, dataset_x, noised_downscaled = create_dataset_from_image(path_to_image, window_size)
  filter_vector = np.where(dataset_x[:, window_size//2, window_size//2] == dataset_y, 0, 1).astype(bool)

  dataset_x = dataset_x[filter_vector]
  dataset_y = dataset_y[filter_vector]

  noised_downscaled = regression_processing(noised_downscaled)
  dataset_x = regression_processing(dataset_x)
  dataset_y = regression_processing(dataset_y)
  return dataset_x, dataset_y, noised_downscaled, filter_vector


if __name__ == "__main__":
  dataset_x, dataset_y = create_dataset_from_image('data/lenna.png')
