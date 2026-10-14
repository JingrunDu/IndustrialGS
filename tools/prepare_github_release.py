"""Create an audited current snapshot without historical large-model Git objects."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SENSITIVE = {'.netrc', '.wandb_auth.json', 'read_token', 'credentials', 'oss_config.json'}
TOKEN_PATTERN = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|LTAI[A-Za-z0-9]{16,})')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true')
    args = parser.parse_args()
    subprocess.run(['git', 'diff', '--check', '--', 'README.md', 'tools', 'docs',
                    'deployment/huanliufa_nebula/submit.sh'], cwd=ROOT, check=True)
    names = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT)
    selected = []
    for name in sorted(set(names.decode().split('\0')) - {''}):
        path = ROOT / name
        if not path.is_file() or path.is_symlink():
            continue
        if path.name in SENSITIVE or path.suffix in {'.env', '.key', '.pem'}:
            raise ValueError('Sensitive filename: ' + name)
        if name.startswith(('data/', 'packages/', 'envs/', 'downloads/', 'backups/', 'result/test1/', 'result/test2/')):
            raise ValueError('Large material selected unexpectedly: ' + name)
        if path.stat().st_size < 2 * 1024**2 and TOKEN_PATTERN.search(path.read_bytes()):
            raise ValueError('Potential credential found in: ' + name)
        if path.stat().st_size > 100 * 1024**2:
            raise ValueError('Unexpected large delivery file: ' + name)
        selected.append(path)
    sizes = Counter()
    for path in selected:
        sizes[path.relative_to(ROOT).parts[0]] += path.stat().st_size
    report = dict(files=len(selected), bytes=sum(sizes.values()), bytes_by_directory=dict(sizes))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.build:
        return
    stage = ROOT / 'backups' / ('github_IndustrialGS_' + time.strftime('%Y%m%d_%H%M%S'))
    stage.mkdir(parents=True)
    for path in selected:
        target = stage / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        target.chmod(path.stat().st_mode & 0o777)
    subprocess.run(['git', 'init'], cwd=stage, check=True)
    subprocess.run(['git', 'symbolic-ref', 'HEAD', 'refs/heads/main'], cwd=stage, check=True)
    subprocess.run(['git', 'remote', 'add', 'origin', 'https://github.com/JingrunDu/IndustrialGS.git'], cwd=stage, check=True)
    subprocess.run(['git', 'lfs', 'install', '--local'], cwd=stage, check=True)
    subprocess.run(['git', 'add', '--all'], cwd=stage, check=True)
    report['stage'] = str(stage)
    report['tree'] = subprocess.check_output(['git', 'write-tree'], cwd=stage, text=True).strip()
    (ROOT / 'backups/github_publication.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('Reviewed snapshot staged:', stage)


if __name__ == '__main__':
    main()
