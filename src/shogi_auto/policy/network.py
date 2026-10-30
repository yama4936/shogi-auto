"""Small policy/value CNN with safe JSON files, including our own warm starts."""
import json
import hashlib
from pathlib import Path

from .features import ACTIONS, PLANES, encode, legal_actions
from ..learning.common import write_json


def build_network():
    import torch
    from torch import nn
    class Residual(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(32, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 32, 3, padding=1)

        def forward(self, x):
            return torch.relu(x + self.conv2(torch.relu(self.conv1(x))))

    class PolicyValue(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Conv2d(PLANES, 32, 3, padding=1)
            self.blocks = nn.Sequential(Residual(), Residual())
            self.policy = nn.Conv2d(32, 27, 1)
            self.value_conv = nn.Conv2d(32, 2, 1)
            self.value_dense = nn.Linear(162, 64)
            self.value_out = nn.Linear(64, 1)

        def forward(self, x):
            trunk = self.blocks(torch.relu(self.stem(x)))
            logits = self.policy(trunk).flatten(1)
            value = torch.relu(self.value_conv(trunk)).flatten(1)
            value = torch.tanh(self.value_out(torch.relu(self.value_dense(value)))).squeeze(1)
            return logits, value
    return PolicyValue()


def weights_digest(network):
    digest = hashlib.sha256()
    for key, value in sorted(network.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def export_network(path, network, metadata):
    state = {key: value.detach().cpu().tolist() for key, value in network.state_dict().items()}
    write_json(path, {'schema': 2, 'architecture': 'pv43-c32-r2', 'planes': PLANES,
                      'actions': ACTIONS, 'state': state, 'metadata': metadata,
                      'weights_sha256': weights_digest(network)})


def load_network(path, device='cpu'):
    import torch
    model = json.loads(Path(path).read_text())
    if (model.get('schema') != 2 or model.get('architecture') != 'pv43-c32-r2' or
            model.get('planes') != PLANES or model.get('actions') != ACTIONS):
        raise ValueError('Unsupported policy/value model')
    network = build_network().to(device)
    expected = network.state_dict()
    if set(model['state']) != set(expected):
        raise ValueError('Unexpected model state keys')
    state = {}
    for key, reference in expected.items():
        value = torch.tensor(model['state'][key], dtype=reference.dtype, device=device)
        if value.shape != reference.shape or not torch.isfinite(value).all():
            raise ValueError(f'Invalid policy/value parameter: {key}')
        state[key] = value
    network.load_state_dict(state)
    if model.get('weights_sha256') != weights_digest(network):
        raise ValueError('Policy/value weights checksum mismatch')
    return network.eval()


class Evaluator:
    def __init__(self, network):
        import torch
        torch.set_num_threads(1)
        self.network = network.cpu().eval()

    def __call__(self, board):
        import torch
        legal = legal_actions(board)
        if not legal:
            return {}, -1.0
        with torch.no_grad():
            logits, value = self.network(torch.from_numpy(encode(board))[None])
            indices = list(legal)
            probabilities = torch.softmax(logits[0, indices], dim=0).tolist()
        return {legal[index]: probability for index, probability in zip(indices, probabilities)}, float(value[0])
