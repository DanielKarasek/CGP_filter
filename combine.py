import cv2
import numpy as np

import image_noising
from cgp_utils import restore_from_saved


def main():
  original_image = cv2.imread("lenna.png")[:, :, 0]
  original_image.squeeze()
  noised_image = image_noising.hurl_noise(0.1, np.copy(original_image))
  noised_image = np.asarray(noised_image, float)
  cv2.namedWindow("output", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
  cv2.namedWindow("original", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
  cv2.resizeWindow("output", 800, 800)
  cv2.resizeWindow("original", 800, 800)
  noised_image /= 255
  filtered_image = restore_from_saved("detect_cgp.pkl", "filter_cgp.pkl", noised_image, 7, 0.55)
  filtered_image = restore_from_saved("detect_cgp.pkl", "filter_cgp.pkl", filtered_image, 7, 0.6)
  cv2.imshow("output", filtered_image)
  cv2.imshow("original", noised_image)
  cv2.waitKey(0)

if __name__ == "__main__":
  main()
