from lib.data.file import DataFile
from lib.games import Game

data = DataFile.open(
    Game.find("chess"),
    r"games\gen_1_games_1723831884211921800",
)

all_game_moves = []
all_game_zero = []
all_game_net = []
all_game_final = []
game_counts = []

game_count = 0

for game in data.simulations:
    with open("list.txt", "r") as f:
        contents = f.read()
    contents = contents.split("\n")
    all_moves = []
    all_zero = []
    all_net = []
    all_final = []

    for pi in game.file_pis:
        positions = data.positions[pi]
        position_zero = [
            positions.zero_v,
            positions.zero_moves_left,
            positions.zero_visits,
            positions.zero_wdl,
        ]
        position_net = [positions.net_v, positions.net_moves_left, positions.net_wdl]
        position_final = [
            positions.final_v,
            positions.final_moves_left,
            positions.final_wdl,
        ]
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
        all_zero.append(position_zero)
        all_net.append(position_net)
        all_final.append(position_final)

    all_moves.pop()

    all_game_moves.append(all_moves)
    all_game_zero.append(all_zero)
    all_game_net.append(all_net)
    all_game_final.append(all_final)
    game_count += 1
    game_counts.append(game_count)


for game_index, (moves, zero_data, net_data, final_data, game_no) in enumerate(
    zip(all_game_moves, all_game_zero, all_game_net, all_game_final, game_counts)
):
    print(f"game {game_no}:")
    for mv, position_zero_data, position_net_data, position_final_data in zip(
        moves, zero_data, net_data, final_data
    ):
        print(
            f"move: {mv}, zero (v, m, n, wdl): {position_zero_data}, net (v, m, wdl): {position_net_data}, final (v, m, wdl): {position_final_data}"
        )
    print()
