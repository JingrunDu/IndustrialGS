"""Run a versioned experiment using the code shipped with the submission."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from environment_archive import DEFAULT_ARCHIVE, restore_environment
from train_node import run_stage


def command_for(config, plan_root, source, output, code, stop_after, resume=None):
    spec = json.loads(Path(config).read_text())
    manifest = plan_root / spec['manifest']
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != spec['manifest_sha256']:
        raise ValueError('Manifest checksum mismatch')
    if len(manifest.read_text().splitlines()) != spec['num_images']:
        raise ValueError('Manifest image count mismatch')
    p = spec['parameters']
    initial = Path(p['init_ply'])
    if not initial.is_absolute():
        p['init_ply'] = str(source / initial)
    if hashlib.sha256(Path(p['init_ply']).read_bytes()).hexdigest() != p['init_ply_sha256']:
        raise ValueError('Initial point cloud checksum mismatch')
    metadata = {'initialization', 'init_ply_sha256', 'camera_extent_policy', 'mask_sigma',
                'wandb_logged_metrics', 'loss_log_interval'}
    command = [str(code / 'train.py'), '--source_path', str(source), '--model_path', str(output),
               '--image_manifest', str(manifest), '--sparse_subdir', 'sparse/0_full',
               '--seed', str(spec['seed']), '--stop_after', str(stop_after),
               '--checkpoint_every', '100000', '--max_gaussians', '6000000']
    for name, value in p.items():
        if name in metadata:
            continue
        if isinstance(value, bool):
            if value:
                command.append('--' + name)
        else:
            command += ['--' + name, *map(str, value if isinstance(value, list) else [value])]
    if resume:
        command += ['--start_checkpoint', str(resume)]
    return spec, command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--stop-after', type=int, default=665000)
    parser.add_argument('--resume', default='')
    parser.add_argument('--output', default='')
    parser.add_argument('--python', default='')
    args = parser.parse_args()
    if args.stop_after < 0:
        raise ValueError('stop-after must be nonnegative; 0 means the full experiment budget')
    if args.experiment not in ['E01', 'E09']:
        raise ValueError('Expected E01 or E09')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('One experiment requires one worker and one GPU')
    here = Path(__file__).resolve().parent
    code = here / 'training_code' if (here / 'training_code').exists() else here.parents[1]
    plan = code / 'experiments'
    config = plan / 'configs' / (args.experiment + '.json')
    spec = json.loads(config.read_text())
    mount = Path(os.environ.get('OSS_MOUNT', '/data/oss_bucket_0'))
    source = Path(os.environ.get('DATA_PATH', str(mount / 'beifen/home/dujingrun.djr/code/3dgs/FastGS/data/perspective')))
    output = Path(args.output) if args.output else mount / spec['output_relative']
    if not args.resume and output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output is not empty; use --resume or a new --output: {output}')
    output.mkdir(parents=True, exist_ok=True)
    python = args.python or os.environ.get('GS_PYTHON') or restore_environment(mount / os.environ.get('ENV_ARCHIVE_REL', DEFAULT_ARCHIVE), Path('/tmp/fmc/fastgs'))
    spec, command = command_for(config, plan, source, output, code, args.stop_after, args.resume)
    env = dict(os.environ)
    env.pop('PYTHONHOME', None)
    env.pop('PYTHONPATH', None)
    env.update(CAMERA_LAZY='1', CAMERA_CACHE_GIB='8', CAMERA_LOAD_WORKERS='8',
               CAMERA_TORCH_THREADS='1', MASK_SIGMA=str(spec['parameters']['mask_sigma']),
               OUTPUT_PATH=str(output), PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1',
               EXPERIMENT_CONFIG=str(config), ENABLE_TENSORBOARD='0')
    env['PATH'] = str(Path(python).parent) + os.pathsep + env.get('PATH', '')
    dependencies = {'plyfile': 'plyfile', 'tqdm': 'tqdm', 'PIL': 'Pillow', 'websockets': 'websockets'}
    if env.get('WANDB_MODE', 'online') != 'disabled':
        dependencies['wandb'] = 'wandb==0.25.0'
    probe = ('import importlib.util,json; print(json.dumps([p for m,p in '
             + repr(dependencies) + '.items() if importlib.util.find_spec(m) is None]))')
    missing = json.loads(subprocess.check_output([python, '-c', probe], env=env, text=True))
    if missing:
        run_stage([python, '-m', 'pip', 'install', '--timeout', '30', '--retries', '2', *missing], '补充辅助依赖', env=env)
    run_stage([python, '-c', 'import torch, fused_ssim, diff_gaussian_rasterization_fastgs, simple_knn._C; '
               'assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'],
              'GPU 与扩展检查', env=env)
    if (here / '.wandb_auth.json').exists():
        env['WANDB_AUTH_FILE'] = str(here / '.wandb_auth.json')
    if args.resume:
        env['TRAIN_RESUME'] = '1'
    (output / 'experiment.json').write_text(json.dumps(spec, indent=2))
    (output / 'command.json').write_text(json.dumps(command, indent=2))
    hashes = {str(p.relative_to(code)): hashlib.sha256(p.read_bytes()).hexdigest()
              for directory in ['arguments', 'scene', 'utils', 'gaussian_renderer']
              for p in (code / directory).rglob('*.py')}
    hashes['train.py'] = hashlib.sha256((code / 'train.py').read_bytes()).hexdigest()
    (output / 'code_hashes.json').write_text(json.dumps(hashes, indent=2))
    print(f'[实验] {args.experiment} seed={spec["seed"]} images={spec["num_images"]} '
          f'预算={spec["parameters"]["iterations"]} 本阶段停止={args.stop_after}\n输出: {output}', flush=True)
    with open(output / 'train.log', 'a', buffering=1) as log:
        process = subprocess.Popen([python, '-u', str(here / 'run_training.py'), *command],
                                   cwd=code, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
        return process.wait()


if __name__ == '__main__':
    sys.exit(main())
