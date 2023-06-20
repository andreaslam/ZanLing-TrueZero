import sqlite3
import gc
gc.disable()
def inserting(games, conn, c):
    for game in games:
        result = game[1]
        moves = game[2]
        print(result)
        print(moves)
        try:
            c.execute("INSERT INTO games (result, moves) VALUES (?,?)",
                    (result, moves))
        except sqlite3.IntegrityError:
            pass

    conn.commit()
    del conn
    del c
    del games
    gc.enable()
    gc.collect()
