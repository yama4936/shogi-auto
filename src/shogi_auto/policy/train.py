"""Bounded, resumable supervised policy/value learning; no self-play RL."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal

from .common import Budget, BudgetExpired, sha256, source_identity, write_json
from .features import ACTIONS, PLANES
from .network import build_network, export_network, weights_digest


def read_data(directory):
    import numpy as np
    directory = Path(directory)
    manifest = json.loads((directory / 'data-manifest.json').read_text())
    if sha256(directory / 'data.npz') != manifest['npz_sha256']:
        raise ValueError('Dataset checksum mismatch')
    with np.load(directory / 'data.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ('x', 'legal', 'policy', 'value', 'split', 'game_id')}
    n = len(arrays['x'])
    if (n == 0 or arrays['x'].shape != (n, PLANES, 9, 9) or arrays['legal'].shape != (n, ACTIONS) or
            arrays['x'].dtype != np.float32 or arrays['value'].dtype != np.float32 or
            arrays['legal'].dtype != np.bool_ or arrays['policy'].shape != (n,) or
            arrays['policy'].dtype.kind not in 'iu' or arrays['value'].shape != (n,) or
            not np.isfinite(arrays['x']).all() or not np.isfinite(arrays['value']).all() or
            np.abs(arrays['value']).max() > 1 or np.any(arrays['policy'] < 0) or np.any(arrays['policy'] >= ACTIONS) or
            not arrays['legal'][np.arange(n), arrays['policy']].all() or
            set(arrays['split'].tolist()) != {'train', 'validation', 'test'}):
        raise ValueError('Invalid policy/value dataset')
    groups = [set(arrays['game_id'][arrays['split'] == split].tolist()) for split in ('train', 'validation', 'test')]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Game IDs overlap dataset splits')
    seen = set()
    for x in arrays['x']:
        key = min(hashlib.sha256(x[:42].tobytes()).hexdigest(),
                  hashlib.sha256(x[:42,:,::-1].copy().tobytes()).hexdigest())
        if key in seen:raise ValueError('Canonical/mirrored board duplicated in dataset')
        seen.add(key)
    return arrays, manifest


def train(directory, data_directory, *, device='cpu', epochs=80, batch_size=128,
          max_seconds=3300, seed=314160, learning_rate=.001, functional_only=False):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    directory, data_directory = Path(directory), Path(data_directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays, manifest = read_data(data_directory)
    counts = manifest['counts']
    if not functional_only and (counts['train'] < 2000 or min(counts['validation'], counts['test']) < 200):
        raise ValueError('GPU/strength preparation requires train>=2000 and validation/test>=200 each')
    if device == 'cuda' and (functional_only or not torch.cuda.is_available()):
        raise ValueError('Real CUDA and sufficient independent data are required; no CPU fallback')
    if not functional_only and manifest.get('features_sha256') != sha256(Path(__file__).with_name('features.py')):
        raise ValueError('Feature implementation differs from the data generator')
    torch.set_num_threads(1);torch.manual_seed(seed)
    if device == 'cuda':torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    model = build_network().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.01)
    settings = {'epochs': epochs, 'batch_size': batch_size, 'seed': seed, 'learning_rate': learning_rate,
                'weight_decay': .01, 'functional_only': functional_only, 'architecture': 'pv43-c32-r2',
                'data_sha256': manifest['npz_sha256'], 'source': source_identity()}
    settings_path = directory / 'training-settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError('Training source/config/data changed; use a new run directory')
    write_json(settings_path, settings)
    tensors = {key: torch.as_tensor(arrays[key], device=device) for key in ('x', 'legal', 'policy', 'value')}
    splits = {split: np.flatnonzero(arrays['split'] == split) for split in ('train', 'validation', 'test')}

    def losses(ids):
        policy, value = model(tensors['x'][ids])
        policy = policy.masked_fill(~tensors['legal'][ids], -1e9)
        ce = torch.nn.functional.cross_entropy(policy, tensors['policy'][ids])
        mse = torch.nn.functional.mse_loss(value, tensors['value'][ids])
        accuracy = (policy.argmax(1) == tensors['policy'][ids]).float().mean()
        return ce, mse, accuracy

    def measure(split):
        totals = np.zeros(3)
        model.eval()
        with torch.no_grad():
            for offset in range(0, len(splits[split]), batch_size):
                ids = torch.as_tensor(splits[split][offset:offset+batch_size], device=device)
                totals += np.asarray([float(x.cpu()) for x in losses(ids)]) * len(ids)
        totals /= len(splits[split])
        return {'policy_ce': float(totals[0]), 'value_mse': float(totals[1]),
                'policy_accuracy': float(totals[2]), 'joint_loss': float(totals[0]+totals[1])}

    metadata = {**settings, 'teacher': {key: manifest[key] for key in
                ('teacher_binary_sha256', 'teacher_weights_sha256', 'teacher_identity')},
                'is_reinforcement_learning': False, 'is_dlshogi': False,
                'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': str(np.__version__),
                'device': device, 'visible_cuda_devices': torch.cuda.device_count(),
                'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                'cuda_runtime': torch.version.cuda}
    initial = directory / 'initial-policy.json'
    best = directory / 'best-policy.json'
    checkpoint = directory / 'checkpoint.pt'
    initial_metrics = {'train': measure('train'), 'validation': measure('validation')}
    start, best_epoch, best_val, logs, cuda_steps = 0, -1, initial_metrics['validation']['joint_loss'], [], 0
    if not initial.exists():export_network(initial, model, metadata)
    if not best.exists():export_network(best, model, {**metadata, 'epoch': -1})
    if checkpoint.exists():
        saved = torch.load(checkpoint, weights_only=True, map_location=device)
        if saved['settings'] != settings:raise ValueError('Checkpoint identity mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        start, best_epoch, best_val, logs = saved['epoch']+1, saved['best_epoch'], saved['best_val'], saved['logs']
        initial_metrics, cuda_steps = saved['initial_metrics'], saved['cuda_steps']
    accounting = directory / 'optimizer-accounting.json'
    if accounting.exists():
        previous = json.loads(accounting.read_text())
        if previous['settings'] != settings:raise ValueError('Optimizer accounting identity mismatch')
        cuda_steps = max(cuda_steps, previous['cuda_steps'])
    budget = Budget(directory, max_seconds, visible_gpus=torch.cuda.device_count() if device == 'cuda' else 0)
    interrupted, launch_steps = False, 0
    def stop(*_):
        nonlocal interrupted
        interrupted = True
    old = signal.signal(signal.SIGTERM, stop)
    def save(epoch):
        temporary = checkpoint.with_suffix('.tmp')
        torch.save({'settings': settings, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'epoch': epoch, 'best_epoch': best_epoch, 'best_val': best_val, 'logs': logs,
                    'initial_metrics': initial_metrics, 'cuda_steps': cuda_steps}, temporary)
        temporary.replace(checkpoint);budget.save()
    status = 'trained'
    try:
        for epoch in range(start, epochs):
            budget.check(reserve=30)
            order = np.random.default_rng(seed+epoch).permutation(splits['train'])
            model.train()
            for offset in range(0, len(order), batch_size):
                # Stop within a batch boundary; the next epoch is replayed on resume
                # only from the last complete epoch's optimizer/model state.
                if interrupted or budget.remaining() <= 30:
                    raise BudgetExpired('Stopped before next batch; resume from last completed epoch')
                ids = torch.as_tensor(order[offset:offset+batch_size], device=device)
                optimizer.zero_grad(set_to_none=True)
                ce, mse, _ = losses(ids)
                (ce+mse).backward();torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step();launch_steps += 1
                if device == 'cuda':
                    torch.cuda.synchronize()
                    cuda_steps += 1
                    write_json(accounting, {'settings': settings, 'cuda_steps': cuda_steps})
            measured = {'epoch': epoch, 'train': measure('train'), 'validation': measure('validation')}
            logs.append(measured)
            if measured['validation']['joint_loss'] < best_val:
                best_val, best_epoch = measured['validation']['joint_loss'], epoch
                export_network(best, model, {**metadata, 'epoch': epoch})
            save(epoch)
            print(f'policy train: epoch {epoch}, validation joint={best_val:.6f}', flush=True)
    except BudgetExpired:
        status = 'paused'
    finally:
        signal.signal(signal.SIGTERM, old);budget.save()
    from .network import load_network
    model = load_network(best, device=device)
    selected = {'train': measure('train'), 'validation': measure('validation')}
    # Test is inspected only for the final selected model after the run completes.
    if status == 'trained':selected['test'] = measure('test')
    report = {**metadata, 'status': status, 'counts': counts, 'initial_metrics': initial_metrics,
              'selected_metrics': selected, 'best_epoch': best_epoch, 'logs': logs,
              'completed_epochs': len(logs), 'optimizer_steps_this_launch': launch_steps,
              'cuda_optimizer_steps_total': cuda_steps,
              'gpu_training_verified': device == 'cuda' and launch_steps > 0,
              'strength_improvement_verified': False, 'functional_only': functional_only,
              'initial_model_sha256': sha256(initial), 'selected_model_sha256': sha256(best),
              'initial_weights_sha256': json.loads(initial.read_text())['weights_sha256'],
              'selected_weights_sha256': weights_digest(model),
              'selected_weights_changed': json.loads(initial.read_text())['weights_sha256'] != weights_digest(model),
              'checkpoint_sha256': sha256(checkpoint) if checkpoint.exists() else None,
              'selection_policy': 'validation policy CE + value MSE only; no arena or test selection',
              'budget': json.loads(budget.path.read_text())}
    write_json(directory / 'training-report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu','cuda'), required=True)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--max-wall-seconds', type=float, default=3300)
    parser.add_argument('--functional-only', action='store_true')
    args = parser.parse_args()
    if not (1 <= args.epochs <= 100 and 1 <= args.batch_size <= 256 and 30 <= args.max_wall_seconds <= 3300):
        raise ValueError('Training limits exceed the bounded first policy experiment')
    print(json.dumps(train(args.run_dir, args.data_dir, device=args.device, epochs=args.epochs,
                           batch_size=args.batch_size, max_seconds=args.max_wall_seconds,
                           functional_only=args.functional_only), ensure_ascii=False), flush=True)


if __name__ == '__main__':main()
