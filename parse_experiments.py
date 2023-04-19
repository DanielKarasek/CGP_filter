import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats

def parse_experiment(experiment_folder: str):
  data = []
  for file_name in os.listdir(experiment_folder):
    keys, values, fitness = parse_file(f"{experiment_folder}/{file_name}")
    data.append([*values, fitness[-1]])
  return pd.DataFrame(data, columns=[*keys, "fitness"])

def parse_experiment_all(experiment_folder: str):
  for file_name in os.listdir(experiment_folder):
    keys, values, fitness, generations = parse_file_all_generations(f"{experiment_folder}/{file_name}")
    break
  data = np.empty((len(keys)+2, 0))
  for file_name in os.listdir(experiment_folder):
    keys, values, fitness, generations = parse_file_all_generations(f"{experiment_folder}/{file_name}")

    data = np.hstack([data, [*values, fitness, generations]])
  print(data.T)
  return pd.DataFrame(data.T, columns=[*keys, "fitness", "generation"])


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

    return keys, values, fitnesses


def parse_file_all_generations(file_name):
  keys, values, fitnesses = parse_file(file_name)
  return keys, np.array([values] * len(fitnesses)).T, fitnesses, np.arange(len(fitnesses))


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
  for i, value in enumerate(values):
    values[i] = float(value)
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

def boxplot_per_column(df: pd.DataFrame):
  df = df.copy(deep=True)
  fig = plt.figure()

  ax = fig.add_subplot(1, 1, 1)
  sns.boxplot(x="columns", y="fitness", data=df, ax=ax)

def boxplot_per_row(df: pd.DataFrame):
  df = df.copy(deep=True)
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  sns.boxplot(x="rows", y="fitness", data=df, ax=ax)

def boxplot_per_L_back(df: pd.DataFrame):
  df = df.copy(deep=True)
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  sns.boxplot(x="levelsback", y="fitness", data=df, ax=ax)

def boxplot_per_mutationrate(df: pd.DataFrame):
  df = df.copy(deep=True)
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  sns.boxplot(x="mutationrate", y="fitness", data=df, ax=ax)

def t_test_mutationrate(df: pd.DataFrame):
  df = df.copy(deep=True)
  df_004 = df[df['mutationrate'] == 0.04]
  df_007 = df[df['mutationrate'] == 0.5]
  res = stats.ttest_ind(df_004['fitness'], df_007['fitness'], equal_var=False)
  return res

def boxplot(df: pd.DataFrame, variable: str):
  df = df.copy(deep=True)
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  sns.boxplot(x=variable, y="fitness", data=df, ax=ax)
  ax.set_title(f"Boxplot of {variable}")
  ax.set_xlabel(variable)
  ax.set_ylabel("Fitness")
  ax.set_ylim(0, 1)

def lineplot(df: pd.DataFrame, variable: str, error_bar):
  df = df.copy(deep=True)
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  pallete = sns.color_palette("bright")
  ax.set_ylim(0, 1)
  # df = df[df["mutationrate"] < 0.15]
  sns.lineplot(x="generation", y="fitness", hue=variable, data=df, ax=ax, palette=pallete, errorbar=error_bar)

def t_test_all(df: pd.DataFrame, variable: str):
  """t_test between all pairs of unique values of variable"""
  df = df.copy(deep=True)
  df.sort_values(by=variable, ascending=False, inplace=True)
  unique_values = pd.unique(df[variable])
  for i, value1 in enumerate(unique_values):
    for j, value2 in enumerate(unique_values):
      df1 = df[df[variable] == value1]
      df2 = df[df[variable] == value2]
      res = stats.ttest_ind(df1['fitness'], df2['fitness'], equal_var=False, alternative='greater')
      print(f"{variable} {value1} {value2} {res}")

def parents():
  df = parse_experiment_all("logs_parents")
  df2 = parse_experiment("logs_parents")
  df["parents"] = df['parents'].astype(int)
  df2["parents"] = df2['parents'].astype(int)
  boxplot(df2, "parents")
  lineplot(df, "parents", error_bar= None)
  t_test_all(df, 'parents')
  plt.show()

def levelsback():
  df = parse_experiment_all("logs_levelsback")
  df2 = parse_experiment("logs_levelsback")
  df["levelsback"] = df['levelsback'].astype(int)
  df2["levelsback"] = df2['levelsback'].astype(int)
  boxplot(df2, "levelsback")
  lineplot(df, "levelsback", error_bar= None)
  t_test_all(df, 'levelsback')
  plt.show()

def mutationrate():
  df = parse_experiment_all("logs_mutationrate")
  df2 = parse_experiment("logs_mutationrate")
  df["mutationrate"] = df['mutationrate'].astype(float)
  df2["mutationrate"] = df2['mutationrate'].astype(float)
  boxplot(df2, "mutationrate")
  lineplot(df, "mutationrate", error_bar= None)
  t_test_all(df, 'mutationrate')
  plt.show()


def columns():
  df = parse_experiment_all("logs_columns_rows")
  df2 = parse_experiment("logs_columns_rows")
  df["columns"] = df['columns'].astype(int)
  df2["columns"] = df2['columns'].astype(int)
  boxplot(df2, "columns")
  lineplot(df, "columns", error_bar= None)
  t_test_all(df, 'columns')
  plt.show()


def rows():
  df = parse_experiment_all("logs_columns_rows")
  df2 = parse_experiment("logs_columns_rows")
  df["rows"] = df['rows'].astype(int)
  df2["rows"] = df2['rows'].astype(int)
  boxplot(df2, "rows")
  lineplot(df, "rows", error_bar= None)
  t_test_all(df, 'rows')
  plt.show()


if __name__ == "__main__":
  parents()
  # levelsback()
  # mutationrate()
  # columns()
  # rows()
  # for file in os.listdir("logs_columns_rows"):
  #   #rename format "n_columns_%xn_rows_%y_repetition_%z.log" to "columns_%x_rows_%y_repetition_%z.log"
  #   os.rename(os.path.join("logs_columns_rows", file),
  #             os.path.join("logs_columns_rows", file.replace("rows", "_rows")))