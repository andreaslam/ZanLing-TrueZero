import os

from paths import data_dir, data_path

print(
    *set(
        [
            data_path(f"python_client_games/{x.split('.')[0]}")
            for x in os.listdir(data_dir("python_client_games"))
        ]
    ),
    sep="\n",
)
