from lib.data.file import DataFile
from lib.games import Game

data = DataFile.open(
    Game.find("chess"),
    r"python_client_games/temp_games_1722456958",
)
for game in data.simulations:
    with open("list.txt", "r") as f:
        contents = f.read()
    contents = contents.split("\n")
    all_moves = []
    for pi in game.file_pis:
        positions = data.positions[pi]
        print(positions.)
        mv = contents[positions.played_mv]
        if positions.move_index % 2 == 1:
            fixed_move = ""
            for c in mv:
                if c.isdigit():
                    fixed_move += str(9 - int(c))
                else:
                    fixed_move += c
            mv = fixed_move
        all_moves.append(mv)

    all_moves.pop()
    for mv in all_moves:
        print(mv)

    print("===")
