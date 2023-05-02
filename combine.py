import cv2
import numpy as np

import image_noising
from cgp_utils import restore_from_saved, filter_noise, load_cgp, restore_image
from image_setup_utils import create_regression_dataset


def just_regression(file_name_prefix: str):
  """
  This function simulated application of regression filter on image and saves and shows the result.
  :param file_name_prefix: prefix of the file name to save the image
  :return:
  """
  cv2.namedWindow("output", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
  cv2.namedWindow("noised", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
  cv2.namedWindow("original", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
  cv2.resizeWindow("output", 800, 800)
  cv2.resizeWindow("noised", 800, 800)
  cv2.resizeWindow("original", 800, 800)

  original_image = cv2.imread("data/lenna_high_res.png")[:, :, 0]
  original_image.squeeze()

  dataset_x, dataset_y, noised_downscaled, filter_vector = create_regression_dataset("data/lenna_high_res.png", 7)
  dataset_x = dataset_x.reshape((dataset_x.shape[0], -1))
  filter_cgp = load_cgp("saved_filters/regression_filters/filter_cgp_3.0.pkl")
  filtered_values = filter_noise(cgp=filter_cgp, sliding_window=dataset_x)
  restored = restore_image(detected_binary_vector=filter_vector,
                           filtered_values_vector=filtered_values,
                           noised_image=noised_downscaled)
  cv2.imshow("output", restored)
  cv2.imshow("noised", noised_downscaled)
  cv2.imshow("original", original_image)
  cv2.imwrite(f"plots_and_images/{file_name_prefix}_regression_output.png", restored*255)
  cv2.imwrite(f"plots_and_images/{file_name_prefix}_regression_noised.png", noised_downscaled*255)
  cv2.imwrite(f"plots_and_images/{file_name_prefix}_regression_original.png", original_image)
  cv2.waitKey(0)


def both(file_name_prefix: str = ""):
  """
  This function simulated application of regression and detection filter on image
  in order to fully restore the image. Images are showed and saved to file.
  :param file_name_prefix: prefix of the file name to save the image to
  :return:
  """
  np.random.seed(35784)
  original_image = cv2.imread("data/lenna_high_res.png")[:, :, 0]
  original_image.squeeze()
  noised_image = image_noising.hurl_noise(0.2, np.copy(original_image))
  noised_image = np.asarray(noised_image, float)

  cv2.namedWindow("output", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
  cv2.namedWindow("noised", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
  cv2.namedWindow("original", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
  cv2.resizeWindow("output", 800, 800)
  cv2.resizeWindow("noised", 800, 800)
  cv2.resizeWindow("original", 800, 800)
  noised_image /= 255

  ## Experimenting with different thresholds and multiple consecutive applications of filters
  filtered_image = restore_from_saved("saved_filters/detection_filters/detect_cgp.pkl",
                                      "saved_filters/regression_filters/filter_cgp_3.0.pkl", noised_image, 7, 0.5)
  filtered_image = restore_from_saved("saved_filters/detection_filters/detect_cgp.pkl",
                                      "saved_filters/regression_filters/filter_cgp_3.0.pkl", filtered_image, 7, 0.48)
  filtered_image = restore_from_saved("saved_filters/detection_filters/detect_cgp.pkl",
                                      "saved_filters/regression_filters/filter_cgp_3.0.pkl", filtered_image, 7, 0.49)


  cv2.imshow("output", filtered_image)
  cv2.imshow("noised", noised_image)

  cv2.imshow("original", original_image)
  cv2.imwrite(f"plots_and_images/{file_name_prefix}_both_output.png", filtered_image*255)
  cv2.imwrite(f"plots_and_images/{file_name_prefix}_both_noised.png", noised_image*255)
  cv2.imwrite(f"plots_and_images/{file_name_prefix}_both_original.png", original_image)

  cv2.waitKey(0)


def apply_median_filter(file_name_prefix: str = ""):
  """
   This function applies median filter on image
  :param file_name_prefix: prefix of the file name to save the image to
  :return:
  """
  np.random.seed(35784)
  from scipy.ndimage import median_filter
  original_image = cv2.imread("data/lenna_high_res.png")[:, :, 0]
  original_image.squeeze()
  noised_image = image_noising.hurl_noise(0.2, np.copy(original_image))
  noised_image = np.asarray(noised_image, float)

  cv2.namedWindow("output", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
  cv2.namedWindow("noised", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
  cv2.namedWindow("original", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
  cv2.resizeWindow("output", 800, 800)
  cv2.resizeWindow("noised", 800, 800)
  cv2.resizeWindow("original", 800, 800)
  noised_image /= 255

  filtered_image = median_filter(noised_image, size=7)

  cv2.imshow("output", filtered_image)
  cv2.imshow("noised", noised_image)

  cv2.imshow("original", original_image)
  cv2.imwrite(f"plots_and_images/{file_name_prefix}_both_output.png", filtered_image * 255)
  cv2.imwrite(f"plots_and_images/{file_name_prefix}_both_noised.png", noised_image * 255)
  cv2.imwrite(f"plots_and_images/{file_name_prefix}_both_original.png", original_image)

  cv2.waitKey(0)


if __name__ == "__main__":
  apply_median_filter("median_filter")
