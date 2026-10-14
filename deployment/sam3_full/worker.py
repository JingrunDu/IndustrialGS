"""Restore small SAM3 environment, cache checkpoint and run one disjoint shard."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run(command, **kwargs):
    started = time.monotonic()
    with subprocess.Popen(command, **kwargs) as proc:
        while True:
            try:
                rc = proc.wait(timeout=30)
                if rc:
                    raise subprocess.CalledProcessError(rc, command)
                return
            except subprocess.TimeoutExpired:
                print(f'[准备/运行] 已用 {time.monotonic()-started:.0f} 秒', flush=True)


def prepare_dependencies(python, env):
    # Preserve the image's torch/CUDA stack. Check all other required imports below.
    requirements = [
        ('timm==1.0.17', 'timm', '1.0.17'),
        ('ftfy==6.1.1', 'ftfy', '6.1.1'),
        ('iopath==0.1.10', 'iopath', '0.1.10'),
        ('numpy>=1.26,<2', 'numpy', None),
        ('scipy==1.14.1', 'scipy', '1.14.1'),
        ('einops', 'einops', None),
        ('huggingface_hub', 'huggingface_hub', None),
        ('pycocotools', 'pycocotools', None),
        ('psutil', 'psutil', None),
        ('setuptools<81', 'setuptools', None),
        ('oss2>=2.18,<3', 'oss2', None),
    ]
    probe = '''
import importlib.metadata as m
import json, sys
from packaging.requirements import Requirement
versions, missing = {}, []
for spec in json.loads(sys.argv[1]):
    req = Requirement(spec)
    try:
        versions[req.name] = m.version(req.name)
        if not req.specifier.contains(versions[req.name]):
            missing.append(spec)
    except m.PackageNotFoundError:
        versions[req.name] = 'missing'
        missing.append(spec)
try:
    import cv2
except ImportError:
    missing.append('opencv-python-headless==4.10.0.84')
print(json.dumps(dict(versions=versions, missing=missing)))
'''
    installed = subprocess.run([python, '-c', probe,
                                json.dumps([req for req, _, _ in requirements])],
                               env=env, text=True, capture_output=True, check=True)
    report = json.loads(installed.stdout)
    print(f'[环境检查] {report["versions"]}', flush=True)
    missing = report['missing']
    if missing:
        print(f'[环境安装] 缺少: {missing}', flush=True)
        run([python, '-m', 'pip', 'install', '--disable-pip-version-check',
             '--timeout', '60', '--retries', '3', *missing], env=env)
    else:
        print('[环境安装] 依赖已存在，跳过 pip install', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, help='Optional explicit shard for standalone runs')
    parser.add_argument('--shards', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output-backend', choices=['filesystem', 'oss'], default='filesystem')
    parser.add_argument('--upload-workers', type=int, default=8)
    parser.add_argument('--upload-queue', type=int, default=64)
    parser.add_argument('--limit', type=int, default=0, help='Smoke-test image count')
    parser.add_argument('--output', help='Separate smoke-test output directory')
    parser.add_argument('--python', help='Use an existing environment instead of creating a venv')
    args = parser.parse_args()
    if args.shard is None:
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        if world_size != args.shards or 'RANK' not in os.environ:
            raise RuntimeError(f'Expected {args.shards} ranks, got WORLD_SIZE={world_size}')
        args.shard = int(os.environ['RANK'])
    elif int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('Explicit --shard is only supported for standalone runs')
    if not 0 <= args.shard < args.shards:
        raise ValueError(f'Invalid shard: {args.shard}/{args.shards}')
    print(f'[分片] {args.shard}/{args.shards}', flush=True)
    root = Path(__file__).resolve().parent
    env = dict(os.environ)
    for key in ['PYTHONHOME', 'PYTHONPATH']:
        env.pop(key, None)
    venv = Path(f'/tmp/sam3_worker_env_{args.shard}')
    if not args.python:
        run([sys.executable, '-m', 'venv', '--system-site-packages', str(venv)], env=env)
    python = args.python or str(venv / 'bin/python')
    prepare_dependencies(python, env)
    env.update(PYTHONPATH=os.environ.get('SAM3_PYTHONPATH', str(root)),
               PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1')
    run([python, '-c', 'import cv2, scipy, psutil, pkg_resources; '
         'from sam3.model_builder import build_sam3_image_model; '
         'from sam3.model.sam3_image_processor import Sam3Processor; '
         'import torch; print("[环境导入通过] torch=", torch.__version__, '
         '"cuda=", torch.version.cuda, "GPU=", torch.cuda.is_available()); '
         'assert torch.cuda.is_available(), "CUDA unavailable"'], env=env)
    # Torch/Triton/CUDA come from pytorch280 image, not the training environment archive.
    checkpoint = Path(f'/tmp/sam3.1_multiplex_{args.shard}.pt')
    source = Path(os.environ['SAM3_CHECKPOINT'])
    print(f'[模型缓存] {source} ({source.stat().st_size} bytes)', flush=True)
    run(['cp', str(source), str(checkpoint)], env=env)
    assert checkpoint.stat().st_size == source.stat().st_size
    command = [python, '-u', str(root / 'generate.py'), '--data', os.environ['DATA_PATH'],
         '--checkpoint', str(checkpoint), '--status-dir', os.environ['STATUS_DIR'],
         '--shard', str(args.shard), '--shards', str(args.shards), '--limit', str(args.limit),
         '--batch-size', str(args.batch_size), '--output-backend', args.output_backend,
         '--upload-workers', str(args.upload_workers), '--upload-queue', str(args.upload_queue)]
    if args.output:
        command.extend(['--output', args.output])
    run(command, env=env)


if __name__ == '__main__':
    main()
