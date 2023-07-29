# https://github.com/Disservin/python-chess-engine/blob/master/src/uci.py
# imports
from sys import stdout
from threading import Thread
import chess
import aiucieval


class UCI:
    def __init__(self) -> None:
        self.out = stdout
        self.board = chess.Board()
        self.thread = None

    def output(self, s) -> None:
        self.out.write(str(s) + "\n")
        self.out.flush()

    def stop(self) -> None:
        try:
            self.thread.join()
        except:
            pass

    def quit(self) -> None:
        try:
            self.thread.join()
        except:
            pass

    def uci(self) -> None:
        self.output("id name Zan1Ling4")
        self.output("id author ALKK")
        self.output("")
        self.output("option name Move Overhead type spin default 5 min 0 max 5000")
        self.output("option name Ponder type check default false")
        self.output("uciok")

    def isready(self) -> None:
        self.output("readyok")

    def ucinewgame(self) -> None:
        pass

    def eval(self) -> None:
        e = aiucieval.static_eval(self.board)
        self.output(e)

    def processCommand(self, input: str) -> None:
        splitted = input.split(" ")
        match splitted[0]:
            case "quit":
                self.quit()
            case "stop":
                self.stop()
            case "ucinewgame":
                self.ucinewgame()
            case "uci":
                self.uci()
            case "isready":
                self.isready()
            case "setoption":
                pass
            case "position":
                fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                movelist = []

                move_idx = input.find("moves")
                if move_idx >= 0:
                    movelist = input[move_idx:].split()[1:]

                if splitted[1] == "fen":
                    position_idx = input.find("fen") + len("fen ")

                    if move_idx >= 0:
                        fen = input[position_idx:move_idx]
                    else:
                        fen = input[position_idx:]

                self.board.set_fen(fen)

                for move in movelist:
                    self.board.push_uci(move)

            case "print":
                print(self.board)
            case "go":
                l = ["depth", "nodes"]

                for limit in l:
                    if limit in splitted:
                        if limit == "depth":
                            depth = int(splitted[splitted.index(limit) + 1])
                            x = aiucieval.analyse_move(self.board, True, depth)
                            self.output(x)
                        elif limit == "nodes":
                            nodes = int(splitted[splitted.index(limit) + 1])
                            x = aiucieval.analyse_move_nodes(self.board, True, nodes)
                        self.output("bestmove "+ x)

                ourTimeStr = "wtime" if self.board.turn == chess.WHITE else "btime"
                ourTimeIncStr = "winc" if self.board.turn == chess.WHITE else "binc"

                if ourTimeStr in input:
                    pass

                if ourTimeIncStr in input:
                    pass

            case "eval":
                return self.eval()
