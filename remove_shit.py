import os


def remove_extra_lines():
  for file_name in os.listdir("logs_n_columns_n_rows"):
    with open(f"logs_n_columns_n_rows/{file_name}", "r") as f:
      lines = f.readlines()
    with open(f"logs_n_columns_n_rows/{file_name}", "w") as f:
      for i in range(60):
        try:
          f.write(lines[i])
        except IndexError:
          break

if __name__ == "__main__":
  remove_extra_lines()
