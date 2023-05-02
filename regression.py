import cgp
import numpy as np
from cgp.individual import IndividualBase

from cgp_utils import filter_noise


class RegressionWrapper:
  """
  Wrapper for regression, this class contains number of functions I tried very briefly as
  error functions, objective function which encapsulates evaluating of a single Individual
  and log function used after end of experiments.
  """
  def __init__(self, dataset_x: np.ndarray, dataset_y: np.ndarray):
    self.dataset_x = dataset_x
    self.dataset_y = dataset_y
    self.dataset_x = self.dataset_x.reshape((self.dataset_x.shape[0], -1))

  def squared_error(self, x: np.ndarray, y: np.ndarray) -> float:
    return np.sum(np.square(x-y))

  def cubic_error(self, x: np.ndarray, y: np.ndarray) -> float:
    return np.sum(np.power(x-y, 3))

  def abs_error(self, x: np.ndarray, y: np.ndarray) -> float:
    return np.sum(np.abs(x-y))

  def mle_gauss(self, x: np.ndarray):
    sigma = np.std(x)
    mu = np.mean(x)
    return mu, sigma

  def kl_error(self, x: np.ndarray, y: np.ndarray) -> float:
    mu_1, sigma_1 = self.mle_gauss(x)
    mu_2, sigma_2 = self.mle_gauss(y)
    if sigma_1 == 0 or sigma_2 == 0:
      return 10000.0
    return np.log(sigma_2/sigma_1) + (sigma_1**2 + (mu_1 - mu_2)**2)/(2*sigma_2**2)

  def final_log_function(self, logger, population: cgp.Population):
    """
    Log function used after end of experiments, it logs number of used functions in the best solution.
    :param logger:
    :param population:
    :return:
    """
    func_dict = population.champion.calculate_count_per_function()
    logger.info(f"Function counts: {func_dict}")

  def objective(self, individual: IndividualBase) -> IndividualBase:
    if not individual.fitness_is_None:
      return individual

    individual.fitness = 0.0

    predicted_y = filter_noise(individual, self.dataset_x)
    error = self.abs_error(predicted_y, self.dataset_y)
    individual.fitness = error/len(self.dataset_y)
    individual.fitness = -individual.fitness

    func_dict = individual.calculate_count_per_function()
    dict_values = [val for val in func_dict.values()]

    # Since I give 7x7 sliding windows, I suppose it be nice to use them,
    # so I punish solutions which use less thea 25, proportionaly
    ratio = 25/np.clip(np.sum(dict_values), 1, 25)
    individual.fitness *= ratio

    return individual


