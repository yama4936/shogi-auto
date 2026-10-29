"""Bounded USI subprocess client for a reviewed, locally built CPU teacher."""
from collections import deque
from pathlib import Path
import queue
import re
import subprocess
import threading
import time

import shogi


class UsiTeacher:
    def __init__(self, binary, eval_dir, log_path, *, hash_mb=64):
        binary, eval_dir = Path(binary).resolve(strict=True), Path(eval_dir).resolve(strict=True)
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open('a')
        self.lines = queue.Queue(maxsize=10000)
        self.recent = deque(maxlen=20)
        self.process = subprocess.Popen([str(binary)], cwd=str(binary.parent), stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        self.reader = threading.Thread(target=self._reader, daemon=True)
        self.reader.start()
        try:
            self.send('usi')
            reply = self.until('usiok', 20)
            self.identity = [line for line in reply if line.startswith('id ')]
            self.options = {re.match(r'option name (.*?) type ', line)[1] for line in reply
                            if re.match(r'option name (.*?) type ', line)}
            required = {'EvalDir': str(eval_dir), 'Threads': '1', 'USI_Hash': str(hash_mb),
                        'BookFile': 'no_book', 'FV_SCALE': '20'}
            for name, value in required.items():
                if name not in self.options:
                    raise ValueError(f'Required teacher option missing: {name}')
                self.send(f'setoption name {name} value {value}')
            self.send('isready')
            self.until('readyok', 20)
            self.new_game()
        except BaseException:
            self.close()
            raise

    def _reader(self):
        try:
            for line in self.process.stdout:
                if len(line) > 100000:
                    break
                self.lines.put(line.rstrip('\n'))
        finally:
            self.lines.put(None)

    def send(self, command):
        if '\n' in command or '\r' in command:
            raise ValueError('USI commands must be single lines')
        self.process.stdin.write(command + '\n')
        self.process.stdin.flush()

    def until(self, prefix, timeout):
        deadline, result = time.monotonic() + timeout, []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f'Teacher waiting for {prefix}; recent={list(self.recent)}')
            try:
                line = self.lines.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(f'Teacher waiting for {prefix}; recent={list(self.recent)}') from None
            if line is None:
                raise RuntimeError(f'Teacher exited early: {list(self.recent)}')
            self.recent.append(line)
            self.log.write(line + '\n')
            self.log.flush()
            result.append(line)
            if line.startswith(prefix):
                return result

    def new_game(self):
        self.send('usinewgame')
        self.send('isready')
        self.until('readyok', 10)

    def analyze(self, board, *, nodes=32768, move_seconds=None, timeout=3):
        # Preserve game history, including repetition, rather than sending only SFEN.
        base = board.copy() if hasattr(board, 'copy') else __import__('copy').deepcopy(board)
        moves = [move.usi() for move in base.move_stack]
        while base.move_stack:
            base.pop()
        position = 'position sfen ' + base.sfen()
        if moves:
            position += ' moves ' + ' '.join(moves)
        self.send(position)
        started = time.monotonic()
        self.send(f'go movetime {max(1, round(move_seconds * 1000))}' if move_seconds is not None else f'go nodes {nodes}')
        try:
            lines = self.until('bestmove ', timeout)
        except TimeoutError:
            self.send('stop')
            self.until('bestmove ', 2)
            raise
        elapsed = time.monotonic() - started
        best = lines[-1].split()[1]
        move = None if best in ('resign', 'win', 'none') else shogi.Move.from_usi(best)
        if move is not None and not board.is_legal(move):
            raise ValueError(f'Teacher returned illegal move: {best}')
        last = None
        for line in lines:
            if line.startswith('info ') and ' score ' in line and (' multipv ' not in line or ' multipv 1 ' in line):
                if ' lowerbound' not in line and ' upperbound' not in line:
                    score = re.search(r' score (cp|mate) (-?\d+)', line)
                    depth = re.search(r' depth (\d+)', line)
                    visited = re.search(r' nodes (\d+)', line)
                    if score and depth and visited:
                        last = {'score_kind': score[1], 'score': int(score[2]),
                                'depth': int(depth[1]), 'nodes': int(visited[1])}
        if last is None:
            raise ValueError('Teacher supplied no exact score/depth/nodes')
        return {**last, 'move': move, 'usi': best, 'seconds': elapsed}

    def close(self):
        if self.process.poll() is None:
            try:
                self.send('quit')
                self.process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()
        self.reader.join(timeout=1)
        self.process.stdin.close()
        self.process.stdout.close()
        self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
