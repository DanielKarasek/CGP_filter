import numpy as np
from cgp.individual import IndividualBase

from cgp_utils import detect_noise


class DetectionWrapper:
  """
  Wrapper for detection by CGP, consists of hamming and f_beta_score which were tested for detection, furthermore
  contains objective function which is used for evaluation of Individuals and final log function which is used
  to log once after experiment.
  """
  def __init__(self, dataset_x: np.ndarray, dataset_y: np.ndarray, beta: float = 1.0):
    self.dataset_x = dataset_x
    self.dataset_y = dataset_y
    window_size = dataset_x.shape[1]
    self.target_detection_vector = np.where(dataset_x[:, window_size//2, window_size//2] == dataset_y, 0, 1)
    self.dataset_x = self.dataset_x.reshape((self.dataset_x.shape[0], -1))
    self.dataset_x = np.asarray(self.dataset_x, dtype=np.float32)
    self.beta = beta

  @staticmethod
  def hamming_distance(x: np.ndarray, y: np.ndarray) -> int:
    return np.sum(np.where(x!=y, 1.0, 0.0))

  def f_beta_score(self, x: np.ndarray, y: np.ndarray) -> float:
    correct_alarm = np.sum(np.where(((x == y) & (y == 1)), 1.0, 0.0))
    false_alarm = np.sum(np.where(x == 1, 1.0, 0.0)) - correct_alarm
    undetected_alarm = np.sum(np.where(y == 1, 1.0, 0.0)) - correct_alarm
    return (1+self.beta)**2 * correct_alarm / ((1+self.beta)**2 * correct_alarm + self.beta**2 * undetected_alarm + false_alarm)

  def objective(self, individual: IndividualBase) -> IndividualBase:
    if not individual.fitness_is_None:
      return individual

    predicted_detection_vector = detect_noise(individual, self.dataset_x)
    f_score = self.f_beta_score(predicted_detection_vector, self.target_detection_vector)

    func_dict = individual.calculate_count_per_function()
    dict_values = [val for val in func_dict.values()]
    ratio = np.clip(np.sum(dict_values), 0, 40)/40
    f_score *= ratio
    individual.fitness = f_score

    return individual

  def final_log_function(self, logger, population):
    """
    Final log function which is used to log once after experiment. Logs number of pixels noised, correctly detected,
    false alarm and missed detection together with function counts of the best solution.
    :param logger:
    :param population:
    :return:
    """
    predicted_detection_vector = detect_noise(population.champion, self.dataset_x)
    ones_in_target = sum(self.target_detection_vector == 1)
    correctly_detected = sum(
      (self.target_detection_vector == predicted_detection_vector) & (predicted_detection_vector == 1))
    false_alarm = sum(predicted_detection_vector == 1) - correctly_detected
    missed_detection = ones_in_target - correctly_detected
    logger.info(f"Total pixels noised: {ones_in_target:>4} Correctly detected: {correctly_detected:>4} "
                f"False alarm: {false_alarm:>4} Missed detection: {missed_detection:>4} ")
    func_dict = population.champion.calculate_count_per_function()
    logger.info(f"Function counts: {func_dict}")


