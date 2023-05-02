import itertools
from typing import Tuple

import numpy as np


def generate_n_random_coordinates_in_range(n: int, x_range: Tuple[float, float], y_range: Tuple[float, float]):
  """
  Generates n random coordinates in the given ranges.
  :param n:
  :param x_range: possible x values range
  :param y_range: possible y values range
  :return:
  """
  cartesian_product = itertools.product(np.arange(x_range[0], x_range[1], 1),
                                        np.arange(y_range[0], y_range[1], 1))
  cartesian_product = np.array(list(cartesian_product))
  coordinates_indices = np.random.choice(np.arange(len(cartesian_product)), n, replace=False)
  return cartesian_product[coordinates_indices].T


def noise_image_at_coordinates(coordinates: np.ndarray, image: np.ndarray):
  """
  Adds noise to the image at the given coordinates.
  :param coordinates: (N,2) array of coordinates
  :param image:
  :return:
  """
  if image.max() > 1.0:
    image[tuple(coordinates[0]), tuple(coordinates[1])] = np.random.randint(0, 255, (coordinates.shape[1], 3))
  else:
    image[tuple(coordinates[0]), tuple(coordinates[1])] = np.random.randint(0, 255, (coordinates.shape[1], 3))/255
  return image


def noise_grayscale_image_at_coordinate(coordinates: np.ndarray, image: np.ndarray):
  """
  Adds noise to the image at the given coordinates.
  :param coordinates: (N,2) array of coordinates
  :param image:
  :return:
  """
  if image.max() > 1.0:
    image[tuple(coordinates[0]), tuple(coordinates[1])] = np.random.randint(0, 255, coordinates.shape[1])
  else:
    image[tuple(coordinates[0]), tuple(coordinates[1])] = np.random.randint(0, 255, coordinates.shape[1])/255
  return image


def hurl_noise(noise_percentage: float, image: np.ndarray):
  """
  Generates number of coordinates corresponding to noise percentage and adds noise to the image.
  :param noise_percentage: 0-1 value of noise percentage
  :param image: input image
  :return: noised image
  """
  image_shape = image.shape

  n = int(noise_percentage * image_shape[0] * image_shape[1])
  coordinates = generate_n_random_coordinates_in_range(n, (0, image_shape[0]), (0, image_shape[1]))
  if len(image.shape) == 3:
    image = noise_image_at_coordinates(coordinates, image)
  else:
    image = noise_grayscale_image_at_coordinate(coordinates, image)
  return image
