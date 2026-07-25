import os

print(
    *set(
        [
            f"./python_client_games/{x.split('.')[0]}"
            for x in os.listdir("python_client_games")
        ]
    ),
    sep="\n",
)
