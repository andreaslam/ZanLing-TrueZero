# code that manually updates path for datafile.txt

import os

from paths import data_dir, data_path

files = os.listdir(data_dir("python_client_games"))

prefix = [f.split(".")[0] for f in files]

prefix = set(prefix)

print(prefix)

with open(data_path("hidden/datafile.txt"), "a") as f:
    f.writelines(data_path("python_client_games/" + file) + "\n" for file in prefix)
