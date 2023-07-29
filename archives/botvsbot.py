import chess
import subprocess, re, sys
sys.setrecursionlimit(100000)
board = chess.Board()

engine_one = subprocess.Popen( # weaker or same engine as stockfish
    "stockfish",
    universal_newlines=True,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    bufsize=1,
)
engine_two = subprocess.Popen( # stronger or same engine as stockfish
    "stockfish",
    universal_newlines=True,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    bufsize=1,
)


def send_move_engine_two(moveorder):
    x = "position startpos moves"
    with open("moves.txt", "r") as f:
        moveorder = f.read()
    # print("\ncmd:\n\t" + x + " " + moveorder)
    engine_two.stdin.write(x + " " + moveorder + "\n")
    get("engine_two")
    # print("\ncmd:\n\t" + "d\n")
    # engine_two.stdin.write("d\n")
    # get("engine_two")
    engine_two.stdin.write("go movetime 3000")
    # print("\ncmd:\n\t" + "go movetime 3000" + "\n")
    output = get("engine_two")
    # print(output)
    actualoutput = output[0]
    pattern = re.compile(r"\w\w?\d\w\w?\d\w?")
    # detect moves
    if actualoutput == "0000":
        finaloutput = "0000"
    else:
        fulloutput = re.findall(pattern, actualoutput)  # not used, saved as data?
        finaloutput = str(re.findall(pattern, actualoutput)[0])
        # print(finaloutput)
    placemove("request from engine_two", finaloutput)


def get(id):
    # using the 'isready' command (engine_one has to answer 'readyok')
    # to indicate current last line of stdout
    if id == "engine_one":
        engine_one.stdin.write("isready\n")
        # print("\ncmd:\n\t" + "isready\n")
        # print("\nengine_one:")
        while True:
            text = engine_one.stdout.readline().strip()
            if text == "readyok":
                # print("REASON: ", text)
                # print("STUCK")
                break
            if text != "":
                print("\t" + text)
            pattern = re.compile(r"bestmove\s\w\d\w\d\sponder\s\w\d\w\d")
            finalmove = re.findall(pattern, text)
            if len(finalmove):
                # print("FOUND IT")
                return re.findall(pattern, text)
    else:
        engine_two.stdin.write("isready\n")
        # print("\ncmd:\n\t" + "isready\n")
        # print("\nengine_one:")
        while True:
            text = engine_two.stdout.readline().strip()
            if text == "readyok":
                # print("REASON: ", text)
                # print("STUCK")
                break
            if text != "":
                print("\t" + text)
            pattern = re.compile(r"bestmove\s\w\d\w\d\sponder\s\w\d\w\d")
            finalmove = re.findall(pattern, text)
            if len(finalmove):
                # print("FOUND IT")
                return re.findall(pattern, text)


def placemove(command, move=""):
    # print("MOVE RECEIVED:", move)
    with open("moves.txt", "a+") as f:
        f.write(move + " ")
        # print("TEXT:", f.read())
    with open("moves.txt", "r") as f:
        moves = f.read().split()
    # print("PREVIOUS MOVES", moves)
    try:
        decision = chess.Move.from_uci(moves[-1])
        board.push(decision)
        # print(board)
        non_stockfish_legal_moves = str(board.legal_moves)
    except Exception:
        non_stockfish_legal_moves = str(board.legal_moves)
    if command == "request from stockfish":
        with open("moves.txt", "r+") as f:
            # print("TEXT AGAIN", f.readlines())
            send_move_engine_one(f.readlines())
    else:
        send_move_engine_two(non_stockfish_legal_moves)


def send_move_engine_one(moveorder):
    x = "position startpos moves"
    with open("moves.txt", "r") as f:
        moveorder = f.read()
    # print("\ncmd:\n\t" + x + " " + moveorder)
    engine_one.stdin.write(x + " " + moveorder + "\n")
    get("engine_one")
    # print("\ncmd:\n\t" + "d\n")
    # engine_one.stdin.write("d\n")
    # get("engine_one")
    engine_one.stdin.write("go movetime 3000")
    # print("\ncmd:\n\t" + "go movetime 3000" + "\n")
    output = get("engine_one")
    # print(output)
    actualoutput = output[0]
    pattern = re.compile(r"\w\w?\d\w\w?\d\w?")
    # detect moves
    if actualoutput == "0000":
        finaloutput = "0000"
    else:
        fulloutput = re.findall(pattern, actualoutput)  # not used, saved as data?
        finaloutput = str(re.findall(pattern, actualoutput)[0])
        # print(finaloutput)
    placemove("request from engine_two", finaloutput)


def starting(first="stockfish"):
    # add data to database
    with open("moves.txt", "r") as f:
        contents = f.read()

    with open("data.txt", "w+") as f1:
        originaltext = f1.read()
        if originaltext == "":
            f1.write(contents)
        else:
            f1.write(originaltext + "\n" + contents)
    # clear data from previous round
    with open("moves.txt", "w") as f2:
        contents = f2.write("")
    
    # starting sequence of engine_one
    starting_sequence_one = [
        "uci",
        "setoption name Hash value 128",
        "setoption name Ponder value false",
        "ucinewgame",
    ]
    get("engine_one")
    for x in starting_sequence_one:
        # print("\ncmd:\n\t" + x)
        engine_one.stdin.write(x + "\n")
        get("engine_one")
        # print("running")
    # print("startup initiated")
    # starting sequence for engine_two
    starting_sequence_two = [
        "uci",
        "setoption name Hash value 128",
        "setoption name Ponder value false",
        "ucinewgame",
    ]
    get("engine_two")
    for x in starting_sequence_two:
        # print("\ncmd:\n\t" + x)
        engine_one.stdin.write(x + "\n")
        get("engine_two")
        # print("running")
    # print("startup initiated")
    if first == "engine_one":
        placemove("request from engine_one")
    else:
        placemove("insert from engine_two")


starting("engine_one")
