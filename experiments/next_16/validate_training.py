"""Local full-dataset smoke and split-vs-continuous resume comparison; no submission."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--densify-check', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    launcher = Path(__file__).resolve().parents[2] / 'deployment/huanliufa_nebula/experiment_node.py'
    env = dict(os.environ, WANDB_MODE='disabled')

    def run(name, stop, resume=None):
        command = [sys.executable, '-u', str(launcher), '--experiment', 'E05', '--python', sys.executable,
                   '--stop-after', str(stop), '--output', str(output / name)]
        if resume:
            command += ['--resume', str(resume)]
        with open(output / f'{name}_{stop}.log', 'w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'[验证] {name} step={stop} passed', flush=True)

    run('split', 1000)
    run('split', 2000, output / 'split/checkpoints/checkpoint_1000.pth')
    run('continuous', 2000)
    split = torch.load(output / 'split/checkpoints/checkpoint_2000.pth', map_location='cpu', weights_only=False)
    continuous = torch.load(output / 'continuous/checkpoints/checkpoint_2000.pth', map_location='cpu', weights_only=False)
    assert split['signature'] == continuous['signature']
    assert split['remaining'] == continuous['remaining']
    assert split['python_rng'] == continuous['python_rng']
    assert torch.equal(split['torch_rng'], continuous['torch_rng'])
    assert all(torch.equal(a, b) for a, b in zip(split['cuda_rng'], continuous['cuda_rng']))
    # CUDA reductions are not guaranteed bitwise deterministic. Report numerical differences.
    differences = {}
    for i, (a, b) in enumerate(zip(split['model'], continuous['model'])):
        if isinstance(a, torch.Tensor):
            differences[str(i)] = float((a - b).abs().max()) if a.numel() else 0
    with open(output / 'split/loss_log.csv') as handle:
        rows = list(csv.DictReader(handle))
    assert [int(r['iteration']) for r in rows] == list(range(100, 2001, 100))
    with open(output / 'continuous/loss_log.csv') as handle:
        other = list(csv.DictReader(handle))
    loss_diff = max(abs(float(a['total_loss']) - float(b['total_loss'])) for a, b in zip(rows, other))
    assert loss_diff < 0.01, loss_diff
    report = dict(status='passed', full_dataset_cameras=90360, split_steps=[1000, 2000],
                  continuous_steps=2000, max_logged_loss_difference=loss_diff,
                  model_max_abs_differences=differences,
                  numerical_tolerance_note='CUDA rasterizer atomic reductions are nondeterministic; loss tolerance 0.01, RNG and sampling exact',
                  runtime_split=json.loads((output / 'split/runtime.json').read_text()),
                  runtime_continuous=json.loads((output / 'continuous/runtime.json').read_text()))
    del split, continuous
    if args.densify_check:
        run('split', 9000, output / 'split/checkpoints/checkpoint_2000.pth')
        run('split', 9100, output / 'split/checkpoints/checkpoint_9000.pth')
        state = torch.load(output / 'split/checkpoints/checkpoint_9100.pth', map_location='cpu', weights_only=False)
        report['after_densification'] = dict(iteration=state['iteration'], gaussians=len(state['model'][1]),
                                           runtime=json.loads((output / 'split/runtime.json').read_text()))
        assert state['iteration'] == 9100
        assert len(state['model'][1]) != 540540
    (output / 'validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
