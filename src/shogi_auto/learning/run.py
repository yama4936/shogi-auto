"""Bounded, resumable experiment CLI. PyTorch is never installed by this code."""
import argparse
import json
import os
from pathlib import Path
import time

from .arena import evaluate_models
from .common import Budget, BudgetExpired, source_identity, write_json
from .data import prepare_data
from .train import train_model

DEFAULTS = {'data_seed': 20261101, 'data_games': 160, 'data_max_plies': 128,
            'samples_per_game': 4, 'teacher_nodes': 4096, 'teacher_min_depth': 2, 'train_seed': 314159,
            'epochs': 200, 'batch_size': 128, 'learning_rate': 0.001,
            'arena_seed': 20261102, 'arena_pairs': 32, 'arena_max_plies': 256,
            'move_seconds': 0.05, 'workers': 2}


def run(directory, config, device='cpu', max_wall_seconds=2700, stage='all'):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; this GPU experiment must not fall back to CPU.')
    if device == 'cuda' and max_wall_seconds > 2700:
        raise ValueError('First GPU run is limited to 2700 wall seconds, reserving setup/save time below 2 allocated GPU-hours on T4 x2.')
    budget = Budget(directory, max_wall_seconds, torch.cuda.device_count() if device == 'cuda' else 0)
    config_path = directory / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Configuration changed; use a new run directory.')
    write_json(config_path, config)
    torch.set_num_threads(1)
    result = {'status': 'running', 'source': source_identity(), 'device': device, 'config': config,
              'gpu_training_verified': False, 'strength_improvement_verified': False}
    started = time.monotonic()
    try:
        arrays, manifest = prepare_data(directory, config, budget)
        result['data'] = manifest
        result['training'] = train_model(directory, arrays, manifest, config, budget, device)
        result['gpu_training_verified'] = result['training']['cuda_optimizer_steps_total'] > 0
        if stage == 'all':
            result['arena'] = evaluate_models(directory, arrays, config, budget)
            result['strength_improvement_verified'] = result['arena']['improvement_evidence']
            result['status'] = 'evaluated'
        else:
            result['status'] = 'trained_not_evaluated'
    except (BudgetExpired, KeyboardInterrupt) as exc:
        result['status'] = 'paused'
        result['reason'] = str(exc)
    except Exception as exc:
        result['status'] = 'failed'
        result['reason'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        budget.save()
        result['this_launch_seconds'] = time.monotonic() - started
        result['budget'] = json.loads(budget.path.read_text())
        write_json(directory / 'run-report.json', result)
    print(json.dumps({'status': result['status'], 'device': device,
                      'gpu_training_verified': result['gpu_training_verified'],
                      'strength_improvement_verified': result['strength_improvement_verified'],
                      'report': str(directory / 'run-report.json')}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--max-wall-seconds', type=float, default=2700)
    parser.add_argument('--stage', choices=['all', 'train'], default='all')
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    config = {**DEFAULTS, **(json.loads(args.config.read_text()) if args.config else {})}
    if not (20 <= config['data_games'] <= 10000 and 1 <= config['samples_per_game'] <= 12 and
            1 <= config['workers'] <= 4 and 1 <= config['arena_pairs'] <= 200 and
            1 <= config['epochs'] <= 1000 and 0.005 <= config['move_seconds'] <= 0.5):
        raise ValueError('Configuration outside supported small-experiment bounds')
    result = run(args.run_dir, config, args.device, args.max_wall_seconds, args.stage)
    if result['status'] == 'paused':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
