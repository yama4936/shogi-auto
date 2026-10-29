"""A fresh, interpretable residual value model; checkpointed supervised learning."""
import json
from pathlib import Path
import platform
import signal
import time

from .common import BudgetExpired, sha256, source_identity, write_json
from .features import FEATURES, SCALE, export_model


def train_model(directory, arrays, data_manifest, config, budget, device='cpu'):
    import numpy as np
    import torch
    directory = Path(directory)
    torch.set_num_threads(1)
    torch.manual_seed(config['train_seed'])
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for the GPU run; do not silently fall back to CPU.')
        torch.cuda.manual_seed_all(config['train_seed'])
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    model = torch.nn.Linear(FEATURES, 1, bias=False)
    with torch.no_grad():
        model.weight.normal_(mean=0, std=0.005)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.1)
    identity = source_identity()
    training_config = {key: config[key] for key in ('train_seed', 'epochs', 'batch_size', 'learning_rate')}
    metadata = {'data_sha256': data_manifest['corpus_sha256'], 'source': identity, 'config': training_config,
                'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__,
                'device': device, 'visible_cuda_devices': torch.cuda.device_count(),
                'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                'cuda_runtime': torch.version.cuda, 'method': 'supervised residual piece-square value, fresh parameters',
                'is_reinforcement_learning': False, 'is_dlshogi': False}
    checkpoint_path = directory / 'checkpoint.pt'
    initial = directory / 'initial-model.json'
    best_path = directory / 'best-model.json'
    if not initial.exists():
        export_model(initial, model.weight.detach().cpu().flatten().tolist(), metadata)
    x = torch.as_tensor(arrays['x'], device=device)
    y = torch.as_tensor(arrays['y'][:, None], device=device)
    masks = {name: torch.as_tensor(arrays['split'] == name, device=device)
             for name in ('train', 'validation', 'test')}
    train_indices = np.flatnonzero(arrays['split'] == 'train')
    def metrics():
        with torch.no_grad():
            predicted = model(x)
            return {name + '_mse': float(((predicted[mask] - y[mask]) ** 2).mean().cpu())
                    for name, mask in masks.items()}
    initial_metrics = metrics()
    start_epoch, best_val, best_epoch, logs = 0, initial_metrics['validation_mse'], -1, []
    cuda_steps_total, launch_steps = 0, 0
    if not best_path.exists():
        export_model(best_path, model.weight.detach().cpu().flatten().tolist(), metadata)
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if saved['data_sha256'] != metadata['data_sha256'] or saved['config'] != training_config or saved['source_sha256'] != identity['source_sha256']:
            raise ValueError('Checkpoint identity mismatch; use a new run directory.')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        start_epoch, best_val, best_epoch = saved['epoch'] + 1, saved['best_val'], saved['best_epoch']
        logs, initial_metrics = saved['logs'], saved['initial_metrics']
        cuda_steps_total = saved['cuda_optimizer_steps_total']
    interrupted = False
    old_handler = None
    if hasattr(signal, 'SIGTERM'):
        def request_stop(*_):
            nonlocal interrupted
            interrupted = True
        old_handler = signal.signal(signal.SIGTERM, request_stop)
    def save_checkpoint(epoch):
        temporary = checkpoint_path.with_suffix('.tmp')
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                    'best_val': best_val, 'best_epoch': best_epoch, 'logs': logs,
                    'initial_metrics': initial_metrics, 'data_sha256': metadata['data_sha256'],
                    'source_sha256': identity['source_sha256'], 'config': training_config,
                    'cuda_optimizer_steps_total': cuda_steps_total}, temporary)
        temporary.replace(checkpoint_path)
        budget.save()
    try:
        for epoch in range(start_epoch, config['epochs']):
            budget.check()
            # Seeded by epoch: exact shuffle order is recoverable without pickled RNG objects.
            order = np.random.default_rng(config['train_seed'] + epoch).permutation(train_indices)
            for offset in range(0, len(order), config['batch_size']):
                ids = torch.as_tensor(order[offset:offset + config['batch_size']], device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.smooth_l1_loss(model(x[ids]), y[ids], beta=0.1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                launch_steps += 1
                if device == 'cuda':
                    cuda_steps_total += 1
            measured = metrics()
            logs.append({'epoch': epoch, **measured})
            if measured['validation_mse'] < best_val:
                best_val, best_epoch = measured['validation_mse'], epoch
                export_model(best_path, model.weight.detach().cpu().flatten().tolist(), {**metadata, 'epoch': epoch})
            save_checkpoint(epoch)
            if epoch % 25 == 0:
                print(f'train: epoch {epoch}, validation MSE={measured["validation_mse"]:.6f}', flush=True)
            if interrupted:
                raise BudgetExpired('Interrupted after a saved epoch; resume from checkpoint.pt')
    finally:
        if old_handler is not None:
            signal.signal(signal.SIGTERM, old_handler)
    chosen = json.loads(best_path.read_text())
    with torch.no_grad():
        model.weight.copy_(torch.as_tensor([chosen['weights']], device=device))
    final_metrics = metrics()
    report = {**metadata, 'initial_metrics': initial_metrics, 'selected_metrics': final_metrics,
              'best_epoch': best_epoch, 'completed_epochs': len(logs), 'logs': logs,
              'initial_model_sha256': sha256(initial), 'selected_model_sha256': sha256(best_path),
              'checkpoint_sha256': sha256(checkpoint_path),
              'optimizer_steps_this_launch': launch_steps, 'cuda_optimizer_steps_total': cuda_steps_total,
              'selection_policy': 'validation MSE only; test metrics and arena outcomes are not used for selection',
              'loss_reduction_is_not_strength_improvement': True}
    write_json(directory / 'training-report.json', report)
    return report
