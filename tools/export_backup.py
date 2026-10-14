"""Inspect or export actual repository files, without Git/LFS caches or credentials."""
import argparse
from collections import Counter
import fnmatch
import hashlib
import json
from pathlib import Path
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {'.git', '__pycache__', '.ipynb_checkpoints', '.nebulactl', 'wandb', 'backups'}
SKIP_NAMES = {'.netrc', '.wandb_auth.json', '.aidata_config', '.aidata_credentials', 'oss_config.json'}


def files():
    for path in sorted(ROOT.rglob('*')):
        rel = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts) or rel.parts[0] in {'data', 'output', 'outputs', 'envs', 'packages', 'downloads', 'downloads_versioned'}:
            continue
        if path.is_symlink() or not path.is_file():
            continue
        if path.name in SKIP_NAMES or any(fnmatch.fnmatch(path.name, pattern) for pattern in ['*.env', '.env*', '*.pyc', '*.pyo', '*.so', '*.pem', '*.key']):
            continue
        if rel.parts[0] == 'submodules' and 'build' in rel.parts:
            continue
        yield path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true')
    args = parser.parse_args()
    selected = list(files())
    groups = Counter()
    size = 0
    for path in selected:
        groups[path.relative_to(ROOT).parts[0]] += path.stat().st_size
        size += path.stat().st_size
    print(json.dumps(dict(files=len(selected), size_bytes=size, size_gib=round(size/1024**3, 3),
                          bytes_by_top_level=dict(groups)), ensure_ascii=False, indent=2))
    if not args.build:
        return
    target = ROOT / 'backups'
    target.mkdir(exist_ok=True)
    name = 'huanliufa_bushu_' + time.strftime('%Y%m%d_%H%M%S')
    archive = target / (name + '.tar')
    partial = target / (name + '.tar.partial')
    entries = []
    with tarfile.open(str(partial), 'w') as tar:
        for index, path in enumerate(selected, 1):
            before = path.stat()
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                prefix = handle.read(256)
                if prefix.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
                    raise ValueError('Run git lfs pull first: %s' % path)
                digest.update(prefix)
                while True:
                    chunk = handle.read(8*1024**2)
                    if not chunk:
                        break
                    digest.update(chunk)
            rel = path.relative_to(ROOT).as_posix()
            tar.add(str(path), arcname='huanliufa_bushu/' + rel, recursive=False)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError('File changed while exporting: %s' % path)
            entries.append(dict(path=rel, size=before.st_size, sha256=digest.hexdigest()))
            if index % 250 == 0:
                print('Archived %d/%d files' % (index, len(selected)), flush=True)
    partial.rename(archive)
    digest = hashlib.sha256()
    with archive.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024**2), b''):
            digest.update(chunk)
    checksum = digest.hexdigest()
    archive.with_suffix('.tar.sha256').write_text(checksum + '  ' + archive.name + '\n')
    archive.with_suffix('.manifest.json').write_text(json.dumps(dict(archive=archive.name, sha256=checksum,
        total_file_bytes=size, files=entries), ensure_ascii=False, indent=2), encoding='utf-8')
    print('Backup:', archive)
    print('SHA256:', checksum)


if __name__ == '__main__':
    main()
