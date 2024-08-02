# code that manually updates path for datafile.txt

import os

files = os.listdir("./python_client_games")

prefix = [f.split(".")[0] for f in files]

prefix = set(prefix)

print(prefix)

with open("datafile.txt", "a") as f:
    for file in prefix:
        f.write("python_client_games/" + file + "\n")
