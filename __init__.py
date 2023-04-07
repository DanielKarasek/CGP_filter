# Daniel Kárásek 2023

import numpy as np
import cv2

from image_noising import hurl_noise


def main():
  image = np.zeros((1920, 1080, 3))
  image = hurl_noise(0.1, image)
  cv2.imshow('target_image', image)
  cv2.waitKey(0)


if __name__ == "__main__":
  main()

