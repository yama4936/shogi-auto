"""Minimal USI adapter; no pondering, clocks, CSA or tournament integration."""
import copy
import sys
import threading

import shogi

from .engine import choose_move


def parse_position(tokens):
    if not tokens:
        raise ValueError("position requires startpos or sfen")
    if tokens[0] == "startpos":
        board, rest = shogi.Board(), tokens[1:]
    elif tokens[0] == "sfen" and len(tokens) >= 5:
        board, rest = shogi.Board(" ".join(tokens[1:5])), tokens[5:]
    else:
        raise ValueError("invalid position")
    if rest:
        if rest[0] != "moves":
            raise ValueError("expected moves")
        for text in rest[1:]:
            move = shogi.Move.from_usi(text)
            if move not in board.legal_moves:
                raise ValueError("illegal move: " + text)
            board.push(move)
    return board


def run(source, target):
    board = shogi.Board()
    worker = None
    stop = threading.Event()
    output_lock = threading.Lock()

    def emit(text):
        with output_lock:
            print(text, file=target, flush=True)

    def finish():
        nonlocal worker
        if worker:
            stop.set()
            worker.join()
            worker = None

    def think(position, seconds, event):
        result = choose_move(position, seconds=seconds, stop=event)
        emit(f"info depth {result.depth} nodes {result.nodes} score cp {result.score}")
        emit("bestmove " + (result.move.usi() if result.move else "resign"))

    for line in source:
        tokens = line.split()
        if not tokens:
            continue
        command = tokens[0]
        try:
            if command == "usi":
                emit("id name shogi-auto CPU baseline")
                emit("id author shogi-auto")
                emit("usiok")
            elif command == "isready":
                emit("readyok")
            elif command in ("usinewgame", "position"):
                finish()
                board = shogi.Board() if command == "usinewgame" else parse_position(tokens[1:])
            elif command == "go":
                finish()
                if "ponder" in tokens or "infinite" in tokens or "mate" in tokens:
                    emit("info string ponder, infinite and go mate are unsupported")
                    emit("bestmove resign")
                    continue
                milliseconds = int(tokens[tokens.index("movetime") + 1]) if "movetime" in tokens else 350
                stop = threading.Event()
                worker = threading.Thread(target=think, args=(copy.deepcopy(board), max(1, milliseconds) / 1000, stop))
                worker.start()
            elif command == "stop":
                finish()
            elif command == "quit":
                break
            elif command == "gameover":
                finish()
            # setoption and unrecognized extension commands are ignored.
        except (ValueError, IndexError) as exc:
            emit("info string " + str(exc))
    finish()


def main():
    run(sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
