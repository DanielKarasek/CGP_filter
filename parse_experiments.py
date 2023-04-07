import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_experiment():
  data = []
  for file_name in os.listdir("logs_n_columns_n_rows"):
    keys, values, fitness = parse_file(f"logs_n_columns_n_rows/{file_name}")
    data.append([*values, fitness])
  return pd.DataFrame(data, columns=[*keys, "fitness"])


def parse_file(file_name):
  keys, values = parse_file_name(file_name)
  with open(file_name, "r") as f:
    lines = f.readlines()
    generations = []
    fitnesses = []
    for i, line in enumerate(lines):
      parsed_line = parse_generation_line(line)
      if parsed_line is False:
        break
      generation, fitness = parsed_line
      generations.append(generation)
      fitnesses.append(fitness)

    return keys, values, fitnesses[59]


def parse_generation_line(line):
  matched = re.match(r'Generation:\s+([0-9]+) Best fitness:\s+([0-9.]+)', line)
  if matched is None:
    return False
  return int(matched.group(1)), float(matched.group(2))


def parse_file_name(file_name):
  file_name_simple = file_name.split("/")[-1]
  file_split = file_name_simple.split("_")
  filtered_split = filter(lambda x: x != "n", file_split)
  filtered_split = list(filtered_split)
  keys = filtered_split[::2][:-1]
  values = filtered_split[1::2][:-1]
  values[0] = int(values[0][:-1])
  values[1] = int(values[1])
  return keys, values


def boxplot_from_columns_rows(df: pd.DataFrame):
  df = df.copy(deep=True)
  df["column_row"] = df["columns"].astype(str) + "x" + df["rows"].astype(str)
  df.sort_values(by="columns", inplace=True)
  unique_columns = pd.unique(df["columns"])
  for unique_column in unique_columns:
    fig = plt.figure()
    fig.suptitle(f"Columns: {unique_column}")
    # create 7x1 subplots
    ax = fig.add_subplot(1, 1, 1)
    col_df = df[df["columns"] == unique_column].copy(deep = True)
    col_df.sort_values(by="rows", inplace=True)

    sns.boxplot(x="rows", y="fitness", data=col_df, ax=ax)
    ax.set_title(f"Columns: {unique_column}")
    ax.set_xlabel("Rows")
    ax.set_ylabel("Fitness")
    ax.set_ylim(0, 1)
    fig.show()
  plt.show()


if __name__ == "__main__":
  df = parse_experiment()
  boxplot_from_columns_rows(df)
