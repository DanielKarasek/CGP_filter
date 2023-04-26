import logging
import os
from itertools import product
from typing import Any, Callable, Dict, List, Tuple

import cv2

from detection import DetectionWrapper
from regression import RegressionWrapper
from image_setup_utils import create_detection_dataset_from_image, create_regression_dataset
from functions import sat_add, sat_sub, cgp_min, cgp_max, greater_than, sat_mul, scale_up, scale_down

from cgp_utils import restore_image

import cgp
import numpy as np
import threading
import time

g_images = []
g_images_changed = False
g_images_changed_lock = threading.Lock()


def image_shower():
  global g_images, g_images_changed
  while True:
    if g_images_changed:
      cv2.destroyAllWindows()
      for name, image in g_images:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 640, 640)
        cv2.imshow(name, image)

      g_images_changed = False
    cv2.waitKey(5000)


def generate_experiments_from_settings(settings: Dict, experiment_names: List[str] = []):
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
    for first_setting, second_setting in product(settings[experiment_varying_vars[0]],
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
                      "n_processes": 14}

    self.evolve_params = {"max_generations": generations, "termination_fitness": termination_fitness}

  def __repr__(self):
    return f"Experiment with experiment values: {self.experimented_values}"

  def __str__(self):
    return f"Experiment with experiment values: {self.experimented_values}"

  def add_end_log_function(self, end_log_function: Callable):
    self.end_log_function = end_log_function

  def _init_logger(self):
    file_path = f"./logs_{self.experiment_type}"
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
    self.population_params["seed"] = np.random.randint(0, 1e7)
    self.pop = cgp.Population(**self.population_params, genome_params=self.genome_params)
    ea = cgp.ea.MuPlusLambda(**self.ea_params)

    cgp.evolve(self.pop,
               self.objective,
               ea,
               **self.evolve_params,
               print_progress=True,
               callback=self.log_generation)

    if self.use_logger:
      self.logger.info(f"Seed: {self.population_params['seed']}")
      self.end_log_function(self.logger, self.pop)
    # dataset_x, dataset_y, noised_downscaled, filter_vector = create_regression_dataset("lenna.png", 7)
    # dataset_x = dataset_x.reshape((dataset_x.shape[0], -1))
    # filtered_values = self.pop.champion.to_numpy()(dataset_x.T)
    # restored = restore_image(detected_binary_vector=filter_vector, filtered_values_vector=filtered_values, noised_image=noised_downscaled)
    # restored += 0.5
    # noised_downscaled += 0.5
    # global g_images, g_images_changed
    #
    # g_images = [("restored", restored), ("orig", noised_downscaled)]
    # g_images_changed = True

    return self.pop


  def run(self, repetitions: int):
    for repetition in range(repetitions):
      self.repetition_id = repetition
      if self.use_logger:
        self.logger = self._init_logger()
      self._run()


def regression_experiments():
  experiment_settings = {
    "parents": 2,
    "window_size": 7,
    "n_outputs": [1],
    "n_columns": 10,  # [25], # 16, 20,
    "n_rows": 10,  # [20, 25], # 6, 8, 12, 14, 16,
    "levelsback": 3,
    "primitives": (sat_add, sat_sub,
                   cgp_min, cgp_max,
                   greater_than, sat_mul, scale_up,
                   scale_down),
    "noffsprings": 25,
    "mutationrate": 0.07,
    "generations": 100,
    "experiment_type": "Regression",
    "termination_fitness": 0.0,
    "use_logger": True}

  dataset_x, dataset_y, noised_downscaled, filter_vector = (
    create_regression_dataset("lenna.png", window_size=experiment_settings["window_size"]))
  regression_wrapper = RegressionWrapper(dataset_x, dataset_y)

  experiment_settings["objective"] = regression_wrapper.objective

  experiments = generate_experiments_from_settings(experiment_settings)

  for experiment in experiments:
    experiment.add_end_log_function(regression_wrapper.final_log_function)
    experiment.run(repetitions=20)


def detection_experiments():
  experiment_settings = {
    "parents": 2,
    "window_size": 5,
    "n_outputs": 1,
    "n_columns": 14,
    "n_rows": 25,
    "levelsback": 6,
    "primitives": (sat_add, sat_sub,
                   cgp_min, cgp_max,
                   greater_than, sat_mul, scale_up,
                   scale_down),
    "noffsprings": 25,
    "mutationrate": 0.07,
    "generations": 250,
    "experiment_type": "Detection",
    "termination_fitness": 1.0,
    "use_logger": True}



  dataset_x, dataset_y = create_detection_dataset_from_image("lenna.png",
                                                             window_size=experiment_settings["window_size"])
  betas = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
  detection_wrappers = [DetectionWrapper(dataset_x, dataset_y, beta) for beta in betas]
  names = ["beta_0.4", "beta_0.6", "beta_0.8", "beta_1.0", "beta_1.2", "beta_1.4"]
  experiment_settings["objective"] = [detection_wrapper.objective for detection_wrapper in detection_wrappers]

  experiments = generate_experiments_from_settings(experiment_settings, experiment_names=names)

  for experiment in experiments:
    experiment.add_end_log_function(detection_wrappers[0].final_log_function)
    experiment.run(repetitions=50)


if __name__ == "__main__":
  threading.Thread(target=image_shower).start()
  # detection_experiments()
  regression_experiments()
