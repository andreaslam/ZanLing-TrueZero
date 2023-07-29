# zan1ling4 | 真零 | (pronounced Jun Ling)
# imports
import sqlite3
import random
import sys
DB_LOCATION = "./agent_traits.db"
conn =  sqlite3.connect(DB_LOCATION)
cursor = conn.cursor()
POPULATION_SIZE = int(sys.argv[1])


# empty and reset DB file before each round

with open(DB_LOCATION, "w") as f:
    pass

conn.execute('''CREATE TABLE IF NOT EXISTS traits 
             (id INTEGER PRIMARY KEY AUTOINCREMENT, lr REAL, layer_size INTEGER, dropout_prob REAL, relu_grad REAL, batch_size INTEGER, n_epochs INTEGER)''')
conn.commit()
conn.close()
category_data = {"small_int": [0,100], "proba": [0.0001,1.0], "big_int": [250,10000], "mid_int":[100,250], "small_float": [1e-10, 5e-3], "mid_float": [1e-5, 5e-1]}

traits = {"lr": "small_float", "layer_size": "big_int", "dropout_prob": "proba", "relu_grad": "mid_float", "batch_size": "big_int", "n_epochs": "mid_int"}

conn_upload = sqlite3.connect("./agent_traits.db")
cursor_upload = conn_upload.cursor()
for _ in range(0,POPULATION_SIZE):
    agent_properties = []
    for trait in traits:
        trait_range = category_data[traits[trait]]
        try:
            rand_value = random.randint(trait_range[0], trait_range[1])
        except ValueError:
            rand_value = random.uniform(trait_range[0], trait_range[1])
        agent_properties.append(rand_value)
    conn_upload.execute("INSERT INTO traits (lr, layer_size, dropout_prob, relu_grad, batch_size, n_epochs) VALUES (?, ?, ?, ?, ?, ?)", (agent_properties[0],agent_properties[1],agent_properties[2],agent_properties[3],agent_properties[4],agent_properties[5]))
conn_upload.commit()
conn_upload.close()
