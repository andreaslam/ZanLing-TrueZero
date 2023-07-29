import sqlite3
import gc
gc.disable()
def inserting(games, conn):
    for game in games:
        for param in game:
            result = param[1]
            moves = param[2]
            try:
                conn.execute("INSERT INTO games (result, moves) VALUES (?,?)",
                        (result, moves))
            except sqlite3.IntegrityError:
                pass

    conn.commit()
    del games
    gc.enable()
    gc.collect()
