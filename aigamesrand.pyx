# 真零 eval engine with NN
# zan1ling4 | 真零 | (pronounced Jun Ling)
# imports
import chess
import chess.pgn
import sqlite3
import tqdm

# create the database and tables if they don't exist
conn = sqlite3.connect("randchess_games.db")
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS games
             (id INTEGER PRIMARY KEY AUTOINCREMENT, result TEXT, moves TEXT UNIQUE)''')
conn.commit()

game_file = ["./random_game.pgn"]

for file in tqdm.tqdm(game_file, desc="files"):
    with open(file, "r+") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            moves = []
            for move in game.mainline_moves():
                board.push(move)
                moves.append(str(move))
            result_map = {"1-0": 1, "0-1": -1, "1/2-1/2": 0, "*":0}
            result = result_map.get(board.result(), None)
            if result is not None:
                try:
                    c.execute("INSERT INTO games (result, moves) VALUES (?, ?)", (result, " ".join(moves)))
                except sqlite3.IntegrityError:
                    # duplicate game detected, skip insertion
                    pass
        # commit after processing all games
            conn.commit()
        conn.close()