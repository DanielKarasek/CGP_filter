import cgp
import numpy as np
from cgp.individual import IndividualBase

from functions import sat_add, sat_sub, cgp_min, cgp_max, sat_mul, negation, greater_than, scale_up, scale_down, const_random
from image_setup_utils import create_detection_dataset_from_image
from cgp_utils import detect_noise


class DetectionWrapper:

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


def experiment_batch(betas, n=50):
  for beta in betas:
    for i in range(n):
      single_experiment(beta, i)


def single_experiment(beta, experiment_id):
  dataset_y, dataset_x = create_detection_dataset_from_image('lenna.png')
  detection_wrapper = DetectionWrapper(dataset_x, dataset_y, beta=beta)

  population_params = {"n_parents": 4, "seed": experiment_id*29}
  genome_params = {"n_inputs": dataset_x.shape[1]**2,
                   "n_outputs": 1,
                   "n_columns": 5,
                   "n_rows": 5,
                   "levels_back": 2,
                   "primitives": (cgp.Add, cgp.Sub, cgp.Mul, cgp.ConstantFloat, sat_add, sat_sub, cgp_min, cgp_max)}
  ea_params = {"n_offsprings": 4,
               "mutation_rate": 0.08,
               "n_processes": 8}

  evolve_params = {"max_generations": 200, "termination_fitness": 1.0}
  pop = cgp.Population(**population_params, genome_params=genome_params)
  ea = cgp.ea.MuPlusLambda(**ea_params)
  history = {"fitness_champion": []}

  def recording_callback(pop):
    history["fitness_champion"].append(pop.champion.fitness)

  cgp.evolve(pop,
             detection_wrapper.objective,
             ea,
             **evolve_params,
             print_progress=True,
             callback=recording_callback)
  f = pop.champion.to_func()
  f = np.vectorize(f)
  predicted_detection_vector = f(*detection_wrapper.dataset_x.T)
  predicted_detection_vector = 1 / (1 + np.exp(-predicted_detection_vector))
  predicted_detection_vector = np.where(predicted_detection_vector > 0.5, 1, 0)
  ones_in_target = sum(detection_wrapper.target_detection_vector == 1)
  correctly_detected = sum(
    (detection_wrapper.target_detection_vector == predicted_detection_vector) & (predicted_detection_vector == 1))
  false_alarm = sum(predicted_detection_vector == 1) - correctly_detected
  missed_detection = ones_in_target - correctly_detected
  with open(f"logs/beta_{beta}_exp_num_{experiment_id}.log", "w+") as log_file:
    log_file.write(f"Total pixels noised: {ones_in_target:>4} Correctly detected: {correctly_detected:>4} "
                   f"False alarm: {false_alarm:>4} Missed detection: {missed_detection:>4}\n")
    for gen, fitness_i in enumerate(history["fitness_champion"]):
      log_file.write(f"Generatio"
                     f"n: {gen:>4} Fitness: {fitness_i:.6f}\n")


def evolve(beta, experiment_id):
  dataset_x, dataset_y = create_detection_dataset_from_image('lenna.png')
  detection_wrapper = DetectionWrapper(dataset_x, dataset_y, beta=beta)


if __name__ == "__main__":
  pass