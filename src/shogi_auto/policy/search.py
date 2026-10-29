"""Bounded PUCT using the same policy/value code before and after learning."""
from dataclasses import dataclass, field
import copy
import math
import time

from ..engine import SearchResult


@dataclass
class Node:
    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    children: dict = field(default_factory=dict)
    expanded: bool = False


def choose_policy_move(board, evaluator, *, seconds=.1, max_simulations=128):
    started = time.monotonic()
    position = copy.deepcopy(board)
    root, deepest, simulations = Node(), 0, 0
    while simulations < max_simulations:
        # Always evaluate the root once; the arena separately measures any overrun.
        if simulations and time.monotonic() - started >= seconds:
            break
        node, path, pushed = root, [root], 0
        try:
            while node.expanded and node.children:
                move, child = max(node.children.items(), key=lambda item:
                    (-item[1].value_sum / item[1].visits if item[1].visits else 0) +
                    1.5 * item[1].prior * math.sqrt(max(1, node.visits)) / (1 + item[1].visits))
                position.push(move)
                pushed += 1
                path.append(child)
                node = child
                if position.is_fourfold_repetition():
                    break
            if position.is_fourfold_repetition():
                value = 0.0
            elif not node.expanded:
                priors, value = evaluator(position)
                node.children = {move: Node(prior=prior) for move, prior in priors.items()}
                node.expanded = True
            else:
                value = -1.0  # Terminal node without legal moves.
            for visited in reversed(path):
                visited.visits += 1
                visited.value_sum += value
                value = -value
            deepest = max(deepest, pushed)
            simulations += 1
        finally:
            for _ in range(pushed):
                position.pop()
    if not root.children:
        return SearchResult(None, 0, simulations, -100000)
    move, selected = max(root.children.items(), key=lambda item: (item[1].visits, item[1].prior))
    return SearchResult(move, deepest, simulations, int(-selected.value_sum / max(1, selected.visits) * 1000),
                        sum(child.visits > 0 for child in root.children.values()), len(root.children), deepest)
