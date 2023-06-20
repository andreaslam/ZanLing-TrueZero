import sqlite3
import multiprocessing
import randomise_process
SRC_DB = '1.db'


def split_tasks(cpu_count, size):
    li = []
    if size > cpu_count:
        non_last_cores = size // cpu_count
        start_counter = 0
        stop_counter = non_last_cores
        for _ in range(cpu_count):
            li.append([start_counter, stop_counter])
            start_counter += non_last_cores
            stop_counter += non_last_cores
        ending = li[-1][-1]
        if ending < size:
            li[-1][-1] = ending + (size - ending)
    else:
        li.append([0, size])
    return li


def manager(cpu):
    size = cpu[1] - cpu[0]
    completed = cpu[0]
    addr = "mini_db" + str(cpu[2]) + ".db"
    print(cpu)
    with open(addr, "w+") as f:
        f.write("")  # empty database by default
    conn = sqlite3.connect(addr)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS games
             (id INTEGER PRIMARY KEY AUTOINCREMENT, result TEXT, moves TEXT UNIQUE)''')
    conn.commit()
    print("BEFORE", size, completed, addr, SRC_DB)
    conn = sqlite3.connect(addr)
    c = conn.cursor()

    # access DB with params
    src_db_conn = sqlite3.connect(SRC_DB)
    src_db_cursor = src_db_conn.cursor()
    print(size, completed)
    src_db_cursor.execute("SELECT * FROM games ORDER BY RANDOM() LIMIT ? OFFSET ?",
                          (size, completed))

    games = src_db_cursor.fetchall()
    print(games)

    randomise_process.inserting(games, conn, src_db_cursor)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    # Connect to the SQLite database
    conn = sqlite3.connect(SRC_DB)
    cursor = conn.cursor()

    # Execute the query to get the row count
    cursor.execute('SELECT COUNT(*) FROM games')
    # Fetch the result
    row_count = cursor.fetchone()[0]
    print("Row count:", row_count)
    conn.close()
    p = int(multiprocessing.cpu_count() * 0.6)  # spare some cpu cores
    print(p)
    load = split_tasks(p, row_count)
    for entry, i_d in zip(load, range(1, p+1)):
        entry.append(i_d)
    with multiprocessing.Pool(p) as pool:
        pool.map(manager, load)
        pool.join()
    # Close the cursor and the database connection
