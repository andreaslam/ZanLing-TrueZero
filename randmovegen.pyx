import chess
import chess.pgn
import random
import tqdm

def generate_moves():
    board = chess.Board()
    while not board.is_game_over():
        move = random.choice(list(board.legal_moves))
        yield move
        board.push(move)

def save_pgn(file_name, moves):
    game = chess.pgn.Game()
    node = game
    for move in moves:
        node = node.add_variation(move)
    with open(file_name, "a+") as f:
        f.write(str(game))


counter = 0
NUM_OF_GAMES = 100000000
for x in tqdm.tqdm(range(0,NUM_OF_GAMES), desc="generating games"):
    moves = generate_moves()
    save_pgn("random_game.pgn", moves)
    counter += 1