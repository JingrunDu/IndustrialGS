"""Run IndustrialGS locally using reviewed configurations, without Nebula or OSS APIs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT = ROOT / 'deployment/huanliufa_nebula'
sys.path.insert(0, str(DEPLOYMENT))
from experiment_node import command_for
from dynamic_update_node import summarize


def build_command(config, data, output, mode, stop_after, resume, additional_steps):
    config, data, output = Path(config).resolve(), Path(data).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new, empty output directory: %s' % output)
    spec = json.loads(config.read_text())
    if spec['parameters'].get('eval', False):
        raise ValueError('This launcher uses all selected images, with eval disabled')
    # Rebase the data location while retaining the reviewed checksum and parameters.
    spec['parameters']['init_ply'] = str(data / 'sparse/0/points3D.ply')
    spec['manifest'] = str((ROOT / 'experiments/next_16' / spec['manifest']).resolve())
    start = 0
    if mode == 'update':
        resume = Path(resume).resolve()
        match = re.fullmatch(r'checkpoint_(\d+)\.pth', resume.name)
        if not match or not resume.is_file() or not Path(str(resume)+'.sha256').is_file():
            raise ValueError('Use checkpoint_<iteration>.pth and its .sha256 file')
        start = int(match.group(1))
        if start <= 0 or additional_steps <= 0:
            raise ValueError('Checkpoint iteration and additional steps must be positive')
        stop_after = start + additional_steps
    if not 0 < stop_after <= spec['parameters']['iterations']:
        raise ValueError('Stopping iteration must be within the original configuration budget')
    for relative in ['images', 'mask_sam3', 'semantic_weights', 'sparse/0_full/images.bin', 'sparse/0_full/cameras.bin']:
        if not (data / relative).exists():
            raise FileNotFoundError(data / relative)
    with tempfile.TemporaryDirectory(prefix='industrialgs_local_config_') as tmp:
        rebased = Path(tmp) / 'config.json'
        rebased.write_text(json.dumps(spec))
        spec, command = command_for(rebased, ROOT / 'experiments/next_16', data, output,
                                    ROOT, stop_after, str(resume) if mode == 'update' else None)
    return spec, command, start, stop_after


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'update'])
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default=str(ROOT / 'experiments/next_16/configs/E01.json'))
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--stop-after', type=int, default=665000)
    parser.add_argument('--resume')
    parser.add_argument('--additional-steps', type=int, default=332500)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.mode == 'update' and not args.resume:
        parser.error('update requires --resume')
    if args.mode == 'train' and args.resume:
        parser.error('Use update mode for checkpoint restoration')
    spec, command, start, stop = build_command(args.config, args.data, args.output, args.mode,
                                              args.stop_after, args.resume, args.additional_steps)
    output = Path(args.output).resolve()
    env = dict(os.environ, OUTPUT_PATH=str(output), TRAIN_ITERATION_OFFSET=str(start),
               CAMERA_LAZY='1', CAMERA_CACHE_GIB=os.environ.get('CAMERA_CACHE_GIB', '8'),
               CAMERA_LOAD_WORKERS=os.environ.get('CAMERA_LOAD_WORKERS', '8'), CAMERA_TORCH_THREADS='1',
               MASK_SIGMA=str(spec['parameters']['mask_sigma']), PYTHONUNBUFFERED='1',
               PYTHONNOUSERSITE='1', ENABLE_TENSORBOARD='0')
    for key in ['PYTHONHOME', 'PYTHONPATH', 'TRAIN_RESUME']:
        env.pop(key, None)
    env['PATH'] = str(Path(args.python).resolve().parent) + os.pathsep + env.get('PATH', '')
    if start:
        env['TRAIN_RESUME'] = '1'
    launch = [args.python, '-u', str(DEPLOYMENT / 'run_training.py'), *command]
    plan = dict(mode=args.mode, source=str(Path(args.data).resolve()), output=str(output),
                start_iteration=start, stop_iteration=stop, new_iterations=stop-start, command=launch)
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return 0
    subprocess.run([args.python, '-c', 'import torch, fused_ssim, diff_gaussian_rasterization_fastgs, simple_knn._C; assert torch.cuda.is_available(), "CUDA unavailable"'], env=env, check=True)
    output.mkdir(parents=True, exist_ok=True)
    config = output / 'experiment.json'
    config.write_text(json.dumps(spec, indent=2))
    env['EXPERIMENT_CONFIG'] = str(config)
    (output / 'command.json').write_text(json.dumps(command, indent=2))
    (output / 'local_run_plan.json').write_text(json.dumps(plan, indent=2))
    paths = [ROOT / 'train.py'] + [p for name in ['arguments', 'scene', 'utils', 'gaussian_renderer'] for p in (ROOT / name).rglob('*.py')]
    (output / 'code_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, indent=2))
    rc = 1
    try:
        with (output / 'train.log').open('w', buffering=1) as log:
            process = subprocess.Popen(launch, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
            with process.stdout:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
            rc = process.wait()
    finally:
        if args.mode == 'update':
            result = summarize(output, rc, start, stop)
            (output / 'dynamic_update_result.json').write_text(json.dumps(result, indent=2))
    return rc if rc else (0 if args.mode == 'train' or result['completed'] else 1)


if __name__ == '__main__':
    sys.exit(main())
