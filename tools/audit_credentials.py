"""Scan publication files and reachable Git history without printing secrets."""
import argparse
import json
from pathlib import Path
import re
import subprocess

PATTERNS = [
    rb'gh[pousr]_[A-Za-z0-9]{30,}',
    rb'github_pat_[A-Za-z0-9_]{50,}',
    rb'LTAI[A-Za-z0-9]{16,}',
    rb'AKIA[A-Z0-9]{16}',
    rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
    rb'(?i)(?:api[_-]?key|access[_-]?key|secret|token|password)[\"\x27]?\s*[:=]\s*[\"\x27]([A-Za-z0-9_+/.=-]{24,})[\"\x27]',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--known-secret-file', action='append', type=Path, default=[])
    args = parser.parse_args()
    known = [p.read_bytes().strip() for p in args.known_secret_file]
    known = [value for value in known if len(value) >= 16]
    patterns = [re.compile(pattern) for pattern in PATTERNS]
    root = args.directory.resolve()
    hits = []

    def scan(data, label):
        if any(p.search(data) for p in patterns) or any(value in data for value in known):
            hits.append(label)

    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    for name in filter(None, names):
        path = root / name
        if path.is_file():
            scan(path.read_bytes(), name)
    objects = subprocess.check_output(['git', 'rev-list', '--objects', '--all'], cwd=root).decode().splitlines()
    blobs = 0
    for line in objects:
        oid = line.split(' ', 1)[0]
        if subprocess.check_output(['git', 'cat-file', '-t', oid], cwd=root).strip() != b'blob':
            continue
        blobs += 1
        scan(subprocess.check_output(['git', 'cat-file', 'blob', oid], cwd=root), 'history:' + oid)
    print(json.dumps(dict(files=len(list(filter(None, names))), history_blobs=blobs,
                         findings=sorted(set(hits))), indent=2))
    if hits:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
