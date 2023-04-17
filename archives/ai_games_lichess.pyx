# 真零 eval engine with NN
# zan1ling4 | 真零 | (pronounced Jun Ling)
# imports
import chess
import chess.pgn
import tqdm
# modifying data
game_file = "./7.pgn"

with open(game_file, "r+") as f:
    text = f.readlines()
    for line in tqdm.tqdm(text, desc="each game"):
        if line[0] == "1":
            # create dummy pgn file
            with open("x.pgn", "w+") as f:
                f.write(line)
            board = chess.Board()
            game = open("x.pgn")
            first_game = chess.pgn.read_game(game)
            for move in first_game.mainline_moves():
                board.push(move)
            result = board.result()
            if result == "1-0":
                with open("ww.pgn", "a+") as f:
                    for x in list(board.move_stack):
                        f.write(str(x) + " ")
                    if line != text[-1]:
                        f.write("\n")
            elif result == "0-1":
                with open("bw.pgn", "a+") as f:
                    for x in list(board.move_stack):
                        f.write(str(x) + " ")
                    if line != text[-1]:
                        f.write("\n")
            elif result == "1/2-1/2":
                with open("drew.pgn", "a+") as f:
                    for x in list(board.move_stack):
                        f.write(str(x) + " ")
                    if line != text[-1]:
                        f.write("\n")