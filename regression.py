import cgp
import cv2
import numpy as np
from cgp.individual import IndividualBase
from scipy.stats import norm

from cgp_utils import filter_noise, restore_image
from functions import sat_add, sat_sub, cgp_min, cgp_max, sat_mul, scale_up, scale_down, negation
from image_setup_utils import create_regression_dataset


class RegressionWrapper:
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

  def final_log_function(self, population, logger):
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
    ratio = 30/np.clip(np.sum(dict_values), 0, 30)
    individual.fitness *= ratio
    return individual




def setup_experiment(n_parents,
                     n_rows,
                     n_columns,
                     levels_back,
                     primitives,
                     n_offsprings,
                     mutation_rate,
                     max_generations):
  population_params = {"n_parents": n_parents, "seed": 0}
  genome_params = {"n_inputs": 49,
                   "n_outputs": 1,
                   "n_columns": n_columns,
                   "n_rows": n_rows,
                   "levels_back": levels_back,
                   "primitives": primitives}
  ea_params = {"n_offsprings": n_offsprings,
               "mutation_rate": mutation_rate,
               "n_processes": 1}

  evolve_params = {"max_generations": max_generations, "termination_fitness": 0}
  pop = cgp.Population(**population_params, genome_params=genome_params)
  ea = cgp.ea.MuPlusLambda(**ea_params)
  return pop, ea, evolve_params



def main():
  dataset_x, dataset_y, noised_full_image, detected_mask_vector = create_regression_dataset('lenna.png')
  regression_wrapper = RegressionWrapper(dataset_x, dataset_y)
  primitives = (sat_mul, sat_add, sat_sub, scale_up, scale_down, cgp_min, cgp_max, negation)

  pop, ea, evolve_params = setup_experiment(n_parents=20,
                                            n_columns=7,
                                            n_rows=7,
                                            levels_back=7,
                                            primitives=primitives,
                                            n_offsprings=50,
                                            mutation_rate=0.15,
                                            max_generations=100)
  history = {"fitness_champion": []}

  def recording_callback(pop):
    history["fitness_champion"].append(pop.champion.fitness)

  cgp.evolve(pop,
             regression_wrapper.objective,
             ea,
             **evolve_params,
             print_progress=True,
             callback=recording_callback)
  filtered_values = filter_noise(pop.champion, regression_wrapper.dataset_x)
  restored_image = restore_image(detected_mask_vector, filtered_values, noised_full_image)
  cv2.namedWindow("noised_unrepaired", cv2.WINDOW_NORMAL)
  cv2.namedWindow("noised_restored", cv2.WINDOW_NORMAL)
  cv2.imshow('noised_unrepaired', noised_full_image.reshape((88, 88)))
  cv2.resizeWindow("noised_unrepaired", 620, 620)
  cv2.imshow('noised_restored', restored_image)
  cv2.resizeWindow("noised_restored", 620, 620)
  cv2.waitKey(0)


if __name__ == "__main__":
  main()
