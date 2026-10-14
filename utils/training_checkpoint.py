"""Complete optimizer/RNG/sampling checkpoints written after optimizer updates."""
import os
from pathlib import Path
import random
import numpy as np
import torch
import tempfile
from utils.artifact_io import publish_file, local_checkpoint

PARAMETERS = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')


def save_checkpoint(path, gaussians, iteration, remaining, ema, background, signature):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(version=2, model=gaussians.capture(gaussians.optimizer_type), iteration=iteration,
                   gradients={name: getattr(gaussians, name).grad for name in PARAMETERS},
                   remaining=remaining, ema=ema, background=background,
                   python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                   torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                   signature=signature)
    with tempfile.TemporaryDirectory(prefix='fastgs_checkpoint_') as temporary:
        local = Path(temporary) / path.name
        torch.save(payload, local)
        publish_file(local, path)
    # Delete only older periodic checkpoints, after the new one is safely closed.
    checkpoints = sorted(path.parent.glob('checkpoint_*.pth'), key=lambda p: int(p.stem.split('_')[-1]))
    for old in checkpoints[:-2]:
        old.unlink()
        Path(str(old) + '.sha256').unlink(missing_ok=True)
    print(f'[断点] 已保存 {path}', flush=True)


def load_checkpoint(path, gaussians, opt, signature):
    with tempfile.TemporaryDirectory(prefix='fastgs_resume_') as temporary:
        local = local_checkpoint(path, temporary)
        state = torch.load(local, weights_only=False)
    if not isinstance(state, dict) or state.get('version') != 2:
        raise ValueError('This experiment requires a complete version-2 checkpoint including accumulated gradients')
    if state['signature'] != signature:
        raise ValueError('Checkpoint dataset/seed/training parameters differ from this run')
    gaussians.restore(state['model'], opt)
    # FastGS updates SH every 16 steps and later both optimizers every 32/64
    # steps. Pending gradients are part of the state, even after optimizer_step().
    for name in PARAMETERS:
        getattr(gaussians, name).grad = state['gradients'][name]
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
    return state
