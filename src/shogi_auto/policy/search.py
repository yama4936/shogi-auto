"""Bounded PUCT using the same policy/value code before and after learning."""
from dataclasses import dataclass, field
import copy
import math
import time

from ..engine import SearchResult, SearchStopped
from ..repetition import repetition_value


@dataclass
class Node:
    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    children: dict = field(default_factory=dict)
    expanded: bool = False


def choose_policy_move(board, evaluator, *, seconds=.1, max_simulations=128, stop=None):
    """Return a legal fallback when a cooperative deadline or stop is reached.

    Evaluators may implement evaluate_with_budget(board, check_budget) to check
    during expensive work. Plain one-argument evaluators remain supported but
    cannot be interrupted inside a call. Neither mode is a hard real-time bound.
    """
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('seconds must be finite and nonnegative')
    if not isinstance(max_simulations, int) or max_simulations < 1:
        raise ValueError('max_simulations must be a positive integer')
    started = time.monotonic()
    deadline = started + seconds
    position = copy.deepcopy(board)
    # Even an expired/cancelled search needs a legal reply, never a false resign.
    fallback = next(iter(position.legal_moves), None)
    if fallback is None:
        return SearchResult(None, 0, 0, -100000)
    root, deepest, simulations = Node(), 0, 0
    longest_simulation = 0.0

    def check_budget():
        if (stop is not None and stop.is_set()) or time.monotonic() >= deadline:
            raise SearchStopped

    evaluate_with_budget = getattr(evaluator, 'evaluate_with_budget', None)
    while simulations < max_simulations:
        node, path, pushed = root, [root], 0
        simulation_started = time.monotonic()
        try:
            check_budget()
            # Leave room for the slowest completed simulation plus 25% margin.
            # This reduces avoidable overruns; unseen stalls remain possible.
            if simulations and deadline - simulation_started < longest_simulation * 1.25:
                break
            while node.expanded and node.children:
                check_budget()
                move, child = max(node.children.items(), key=lambda item:
                    (-item[1].value_sum / item[1].visits if item[1].visits else 0) +
                    1.5 * item[1].prior * math.sqrt(max(1, node.visits)) / (1 + item[1].visits))
                position.push(move)
                pushed += 1
                path.append(child)
                node = child
                if position.is_fourfold_repetition():
                    break
            check_budget()
            if position.is_fourfold_repetition():
                value = repetition_value(position, check_budget)
            elif not node.expanded:
                if evaluate_with_budget is None:
                    priors, value = evaluator(position)
                else:
                    priors, value = evaluate_with_budget(position, check_budget)
                check_budget()
                children = {}
                for move, prior in priors.items():
                    check_budget()
                    children[move] = Node(prior=prior)
                # Commit expansion only when it finished within the budget.
                node.children = children
                node.expanded = True
            else:
                value = -1.0  # Terminal node without legal moves.
            for visited in reversed(path):
                visited.visits += 1
                visited.value_sum += value
                value = -value
            deepest = max(deepest, pushed)
            simulations += 1
        except SearchStopped:
            break
        finally:
            for _ in range(pushed):
                position.pop()
        longest_simulation = max(longest_simulation, time.monotonic() - simulation_started)
    if not root.children:
        return SearchResult(fallback, 0, simulations, 0, 0, 0, 0)
    move, selected = max(root.children.items(), key=lambda item: (item[1].visits, item[1].prior))
    return SearchResult(move, deepest, simulations, int(-selected.value_sum / max(1, selected.visits) * 1000),
                        sum(child.visits > 0 for child in root.children.values()), len(root.children), deepest)
