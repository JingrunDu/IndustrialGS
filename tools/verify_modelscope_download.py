"""Download committed delivery artifacts into a fresh verification workspace."""
import argparse
import json
from pathlib import Path

from modelscope_delivery import WORK, STAGE, REPO, ENVIRONMENT, api_client, sha, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('section', choices=['data/subset_6650', 'environment', 'result/test1/modeling'])
    parser.add_argument('--extras-only', action='store_true')
    args = parser.parse_args()
    api = api_client()
    directory = STAGE / args.section
    if args.section == 'environment':
        directory = directory / ENVIRONMENT
    paths = [p for p in directory.rglob('*') if p.is_file()]
    if args.extras_only:
        if args.section != 'environment':
            parser.error('--extras-only requires environment')
        paths = [p for p in paths if 'wheels' in p.parts or p.name == 'requirements-runtime.txt']
    if args.section.startswith('result/'):
        paths = [p for p in paths if 'checkpoints' in p.parts]
    downloaded = []
    for source in sorted(paths):
        name = source.relative_to(STAGE).as_posix()
        digest = sha(source)
        target = api.download_file(REPO, 'dataset', name, revision='master',
                                   local_dir=WORK / 'downloads', expected_sha256=digest)
        if sha(target) != digest:
            raise ValueError('Downloaded checksum differs: %s' % name)
        downloaded.append(dict(path=name, sha256=digest, size=target.stat().st_size))
        print('Download verified:', name, flush=True)
    suffix = '_extras' if args.extras_only else ''
    write_json(WORK / ('download_verified_' + args.section.replace('/', '_') + suffix + '.json'), downloaded)


if __name__ == '__main__':
    main()
