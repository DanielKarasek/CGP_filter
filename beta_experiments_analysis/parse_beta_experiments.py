import os
import re
from typing import Tuple
import pandas as pd


def parse_name(file_name: str) -> float:
  re_match = re.match(r"beta_([0-9.]+)", file_name)
  return float(re_match.group(1))


def parse_file(file_name):
  with open(file_name, "r") as f:
    init_line = f.readline()
    match = re.match(r'Total pixels noised:\s+([0-9]+) '
                     r'Correctly detected:\s+([0-9]+) '
                     r'False alarm:\s+([0-9]+) '
                     r'Missed detection:\s+([0-9]+)', init_line)
    total_pixels_noised, correctly_detected, false_alarm, missed_detection = match.groups()
    total_pixels_noised, correctly_detected = int(total_pixels_noised), int(correctly_detected)
    false_alarm, missed_detection = int(false_alarm), int(missed_detection)
    lines = f.readlines()
    for line in lines:
      match = re.match(r'Generation:\s+([0-9]+) Fitness:\s+([0-9.]+)', line)

    gen, fitness = match.groups()
    gen, fitness = int(gen), float(fitness)
    return total_pixels_noised, correctly_detected, false_alarm, missed_detection, fitness


def parse_files() -> Tuple[pd.DataFrame, int]:
  data = []
  for file_name in os.listdir("../logs"):
    argument = parse_name(file_name)

    (total_pixels_noised, correctly_detected,
     false_alarm, missed_detection, fitness) = parse_file(f"../logs/{file_name}")
    data.append([argument, correctly_detected, false_alarm,
                  missed_detection, fitness])
  return pd.DataFrame(data, columns=["beta", "correctly_detected",
                                     "false_alarm", "missed_detection",
                                     "fitness"]), total_pixels_noised


if __name__ == "__main__":
  df, total_pixels_noised = parse_files()
