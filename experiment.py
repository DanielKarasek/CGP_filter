import logging
import os
import pickle
from typing import Any, Callable, Dict, List, Tuple

import cgp
import numpy as np
from tqdm import tqdm

from detection import DetectionWrapper
from functions import sat_add, sat_sub, cgp_min, cgp_max, greater_than, sat_mul, scale_up, scale_down
from image_setup_utils import create_detection_dataset_from_image, create_regression_dataset
from regression import RegressionWrapper


def generate_experiments_from_settings(settings: Dict, experiment_names: List[str] = []):
  """
  This function generates a list of experiments from a dictionary of settings.
  :param settings: Dictionary of settings, if any entry is a List it is then considered to be
                   a parameter that should be varied. Experiments are generated for all combinations
                   of varied parameters (max 2 params can vary at the same time)
                   for example {"generations: [10, 20], "population_size": [100, 200]} will generate 4 experiments
                   [(10 generations, 100 population),
                    (10 generations, 200 population),
                    (20 generations, 100 population),
                    (20 generations, 200 population)]
  :param experiment_names: list of names for each experiment, must be equal to total number of experiments
  :return: Generator for different experiments
  """
  experiment_varying_vars = []
  for key, value in settings.items():
    if isinstance(value, list):
      experiment_varying_vars.append(key)
      print(key)
  assert (len(experiment_varying_vars) < 3 or
          len(experiment_varying_vars) > 0 or
          "No more than two parameters should be varied at the same time")
  varying_params = []
  if len(experiment_varying_vars) == 2:
    # Either we want all AxB combination or just pairs
    # for first_setting, second_setting in product(settings[experiment_varying_vars[0]],
    #                                              settings[experiment_varying_vars[1]]):
    for first_setting, second_setting in zip(settings[experiment_varying_vars[0]],
                                             settings[experiment_varying_vars[1]]):
       varying_params.append({experiment_varying_vars[0]: first_setting,
                              experiment_varying_vars[1]: second_setting})
  elif len(experiment_varying_vars) == 1:
    varying_params = [{experiment_varying_vars[0]: setting} for
                      setting in settings[experiment_varying_vars[0]]]
  for key in experiment_varying_vars:
    del settings[key]
  if experiment_names:
    experiments = (Experiment(**settings,
                              **varying_params_group,
                              experiment_name=experiment_names[i],
                              experimented_values=varying_params_group) for
                   i, varying_params_group in enumerate(varying_params))
  else:
    print("No experiment names provided, using default name")
    experiments = (Experiment(**settings,
                              **varying_params_group,
                              experimented_values=varying_params_group) for
                   varying_params_group in varying_params)

  return experiments


class Experiment:
  """
  Experiment wrapper, wraps all the parameters for multiple experiment runs + functionality to log results +
  show results.
  """
  def __init__(self, parents: int, window_size: int, n_outputs: int, n_columns: int,
               n_rows: int, levelsback: int, primitives: Tuple, noffsprings: int,
               mutationrate: float, generations: int, termination_fitness: float,
               experimented_values: Dict[str, Any], objective: Callable, use_logger: bool = True,
               experiment_name: str = "", experiment_type: str = "regression", *args, **kwargs):
    self.experiment_type = experiment_type
    self.experiment_name = experiment_name
    self.repetition_id = 0
    self.end_log_function = lambda exp_logger, pop: None
    self.objective = objective
    self.experimented_values = experimented_values
    self.use_logger = use_logger
    self.population_params = {"n_parents": parents, "seed": np.random.randint(0, 1e7)}
    self.genome_params = {"n_inputs": window_size**2,
                          "n_outputs": n_outputs,
                          "n_columns": n_columns,
                          "n_rows": n_rows,
                          "levels_back": levelsback,
                          "primitives": primitives}
    self.ea_params = {"n_offsprings": noffsprings,
                      "mutation_rate": mutationrate,
                      "n_processes": 16}
    self.max_generations = generations

  def __repr__(self):
    return f"Experiment with experiment values: {self.experimented_values}"

  def __str__(self):
    return f"Experiment with experiment values: {self.experimented_values}"

  def add_end_log_function(self, end_log_function: Callable):
    self.end_log_function = end_log_function

  def _init_logger(self):
    """Inits logger for experiment and n_th repetition"""
    file_path = f"./all_logs/logs_{self.experiment_type}"
    for key in self.experimented_values.keys():
      file_path = f"{file_path}_{key}"
    try:
      os.mkdir(file_path)
    except FileExistsError as e:
      pass
    file_path = f"{file_path}/"
    if self.experiment_name and len(self.experimented_values.items()) > 0:
      file_path = f"{file_path}{self.experiment_name}"
    else:
      for key, value in self.experimented_values.items():
        file_path = f"{file_path}{key}_{value}"
    file_path = f"{file_path}_repetition_{self.repetition_id}"

    experiment_logger = logging.getLogger("experiment_logger")
    for handler in experiment_logger.handlers[:]:
      experiment_logger.removeHandler(handler)
    file_handler = logging.FileHandler(f"{file_path}.log", mode="w")
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    experiment_logger.addHandler(file_handler)
    experiment_logger.setLevel(logging.INFO)

    print(f"Logging to {file_path}.log")

    return experiment_logger

  def log_generation(self, pop: cgp.Population):
    if self.use_logger:
      best_fitness = pop.champion.fitness
      self.logger.info(f"Generation: {self.pop.generation} Best fitness: {best_fitness}")

  def _run(self) -> cgp.Population:
    self.evolve_loop()

    if self.use_logger:
      self.logger.info(f"Seed: {self.population_params['seed']}")
      self.end_log_function(self.logger, self.pop)
    return self.pop

  def evolve_loop(self):
    """
    Performs one run of evolution, little trick with increase of mutation_rate and parent count when no improvement
    is made for every (n_offspring *10) generations. This should improve chance of getting out of strong local optima
    into better neighborhood.
    """
    self.population_params["seed"] = np.random.randint(1e7)
    self.pop = cgp.Population(**self.population_params, genome_params=self.genome_params)
    ea = cgp.ea.MuPlusLambda(**self.ea_params)

    last_best = -99999
    last_improvement_step = 0

    cgp.evolve(self.pop,
               self.objective,
               ea)
    self.log_generation(self.pop)
    mutation_rate_increase_accumulator = 0
    with tqdm(total=100) as pbar:
      for i in np.arange(self.max_generations):
        cgp.hl_api.evolve_continue(self.pop,
                                   self.objective,
                                   ea)
        self.log_generation(self.pop)
        self.pop.generation = 0
        mutation_rate_increase_accumulator += 1
        if self.pop.champion.fitness > last_best:
          last_best = self.pop.champion.fitness
          last_improvement_step = i

          ea._mutation_rate = self.ea_params["mutation_rate"]
          self.pop.n_parents = self.population_params["n_parents"]

          mutation_rate_increase_accumulator = 0

        if mutation_rate_increase_accumulator > self.max_generations//70:
          ea._mutation_rate = np.clip(ea._mutation_rate * 1.1, 0, self.ea_params["mutation_rate"]*1.6)
          self.pop.n_parents = np.clip(self.pop.n_parents+1, 0, min(self.ea_params["n_offsprings"], 8))
          mutation_rate_increase_accumulator = 0

        pbar.update(100 / self.max_generations)
        pbar.set_description(f"Generation {i} Last improvement: {last_improvement_step}")
        pbar.set_postfix_str(f"Best fitness: {last_best}, Mutation rate: {ea._mutation_rate:.2f}, Parents: {self.pop.n_parents}")
    with open(f"regression_actual_best_{self.experiment_name}.pkl", "wb") as f:
      pickle.dump(self.pop.champion, f)


  def run(self, repetitions: int):
    for repetition in range(repetitions):
      self.repetition_id = repetition
      if self.use_logger:
        self.logger = self._init_logger()
      self._run()


def regression_experiments():
  """
  Regression experiment setup
  :return:
  """
  experiment_settings = {
    "parents": 2,
    "window_size": 7,
    "n_outputs": 1,
    "n_columns": 14,
    "n_rows": 25,
    "levelsback": 6,
    "primitives": (sat_add, sat_sub,
                   cgp_min, cgp_max,
                   greater_than, sat_mul, scale_up,
                   scale_down),
    "noffsprings": 6,
    "mutationrate": 0.07,
    "generations": 3000,
    "experiment_type": "Regression",
    "termination_fitness": 0.0,
    "use_logger": [True]}

  dataset_x, dataset_y, noised_downscaled, filter_vector = (
    create_regression_dataset("data/lenna.png", window_size=experiment_settings["window_size"]))
  regression_wrapper = RegressionWrapper(dataset_x, dataset_y)

  experiment_settings["objective"] = regression_wrapper.objective

  experiments = generate_experiments_from_settings(experiment_settings)

  for experiment in experiments:
    experiment.add_end_log_function(regression_wrapper.final_log_function)
    experiment.run(repetitions=20)


def detection_experiments():
  """Detection experiment setups"""
  experiment_settings = {
    "parents": 2,
    "window_size": 7,
    "n_outputs": 1,
    "n_columns": 14,
    "n_rows": 25,
    "levelsback": 6,
    "primitives": (sat_add, sat_sub,
                   cgp_min, cgp_max,
                   greater_than, sat_mul, scale_up,
                   scale_down),
    "noffsprings": 6,
    "mutationrate": 0.07,
    "generations": 12000,
    "experiment_type": "Detection",
    "termination_fitness": 1.0,
    "use_logger": True}



  dataset_x, dataset_y = create_detection_dataset_from_image("data/lenna.png",
                                                             window_size=experiment_settings["window_size"])
  # betas = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
  # detection_wrappers = [DetectionWrapper(dataset_x, dataset_y, beta) for beta in betas]
  # names = ["beta_0.4", "beta_0.6", "beta_0.8", "beta_1.0", "beta_1.2", "beta_1.4"]
  # experiment_settings["objective"] = [detection_wrapper.objective for detection_wrapper in detection_wrappers]
  detection_wrapper = DetectionWrapper(dataset_x, dataset_y, beta=1.2)

  experiment_settings["objective"] = detection_wrapper.objective
  experiments = generate_experiments_from_settings(experiment_settings)

  for experiment in experiments:
    experiment.add_end_log_function(detection_wrapper.final_log_function)
    experiment.run(repetitions=20)


if __name__ == "__main__":
  # detection_experiments()
  regression_experiments()
