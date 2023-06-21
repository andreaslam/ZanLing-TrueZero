import sqlite3
import random
import multiprocessing
import randomise_process
import tqdm

SRC_DB = "1.db"


def split_tasks(cpu_count, size):
    li = []
    random_idx = random.sample(range(1, size + 1), size)
    if size > cpu_count:
        non_last_cores = size // cpu_count
        start_counter = 0
        stop_counter = non_last_cores
        for sub in range(cpu_count):
            li.append(random_idx[start_counter:stop_counter])
            start_counter += non_last_cores
            stop_counter += non_last_cores
            if (
                sub == cpu_count - 2
            ):  # fit everything in the last cpu core, since the range() ends one less than the actual, so everything goes to the "penultimate" (last if counting from 0)
                li.append(random_idx[start_counter:])
                break
    else:
        li.append([0, size])
    return li


def manager(cpu):
    addr = "mini_db" + str(cpu[-1]) + ".db"
    with open(addr, "w+") as f:
        f.write("")  # empty database by default
    conn = sqlite3.connect(addr)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS games
             (id INTEGER PRIMARY KEY AUTOINCREMENT, result TEXT, moves TEXT UNIQUE)"""
    )
    conn.commit()
    conn = sqlite3.connect(addr)
    c = conn.cursor()

    # access DB with params
    src_db_conn = sqlite3.connect(SRC_DB)
    src_db_cursor = src_db_conn.cursor()
    cpu = cpu[:-1] # remove last item which was used for naming the mini databases
    games = []
    for index in tqdm.tqdm(cpu, desc="each game"):
        src_db_cursor.execute("SELECT * FROM games where id=?", (index,))
        games.append(src_db_cursor.fetchall())
    randomise_process.inserting(games, conn)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    # Connect to the SQLite database
    conn = sqlite3.connect(SRC_DB)
    cursor = conn.cursor()

    # Execute the query to get the row count
    cursor.execute("SELECT COUNT(*) FROM games")
    # Fetch the result
    row_count = cursor.fetchone()[0]
    conn.close()
    p = int(multiprocessing.cpu_count() * 0.6)  # spare some cpu cores
    load = split_tasks(p, row_count)
    for entry, i_d in zip(load, range(1, p + 1)):
        entry.append(i_d)
    with multiprocessing.Pool(p) as pool:
        pool.map(manager, load)
    pool.join()
    # Close the cursor and the database connection
