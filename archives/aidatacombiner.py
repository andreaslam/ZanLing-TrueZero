import sqlite3
import os

db_src = os.listdir(os.getcwd())
db_src = [x for x in db_src if x[-3:] == ".db"]
NEW_DB_LOCATION = "all_data.db"
new_db_conn = sqlite3.connect(NEW_DB_LOCATION)
c = new_db_conn.cursor()
c.execute(
    """CREATE TABLE IF NOT EXISTS games
             (id INTEGER PRIMARY KEY AUTOINCREMENT, result TEXT, moves TEXT UNIQUE)"""
)

new_db_conn.commit()
try:
    db_src.remove(NEW_DB_LOCATION)
except Exception:
    pass
print(db_src)


def merge_dbs(source, dest):
    source.execute("SELECT * FROM games")
    games = source.fetchall()
    for game in games:
        result = game[1]
        moves = game[2]
        try:
            dest.execute(
                "INSERT INTO games (result, moves) VALUES (?, ?)", (result, moves)
            )
        except sqlite3.IntegrityError:
            # duplicate game detected, skip insertion
            pass
    new_db_conn.commit()
    source.close()


for db in db_src:
    source_conn = sqlite3.connect(db)
    cs = source_conn.cursor()
    merge_dbs(cs, c)
    print(db)
new_db_conn.close()
