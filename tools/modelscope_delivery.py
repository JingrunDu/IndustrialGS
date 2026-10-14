"""Build, upload and verify reproducible IndustrialGS delivery artifacts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'backups/modelscope_delivery'
STAGE = WORK / 'payload'
DATA = Path('/data/oss_bucket_0/beifen/home/dujingrun.djr/code/3dgs/FastGS/data/perspective')
REPO = 'JingrunDu/converter_value'
ENVIRONMENT = 'industrialgs_py310_cu121_v1'
LIMIT = 2 * 1024**3


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def camera_names(path):
    with path.open('rb') as handle:
        count = struct.unpack('<Q', handle.read(8))[0]
        names = []
        for _ in range(count):
            handle.read(64)
            name = bytearray()
            while True:
                char = handle.read(1)
                if not char:
                    raise ValueError('Truncated camera model')
                if char == b'\0':
                    break
                name.extend(char)
            points = struct.unpack('<Q', handle.read(8))[0]
            handle.seek(points * 24, 1)
            names.append(name.decode())
    if len(set(names)) != count:
        raise ValueError('Duplicate camera names')
    return sorted(names)


class HashReader:
    def __init__(self, handle):
        self.handle = handle
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        chunk = self.handle.read(size)
        self.digest.update(chunk)
        return chunk


def create_shard(dataset, number, names):
    from PIL import Image
    directory = STAGE / 'data' / dataset
    archive = directory / ('samples-%03d.tar' % number)
    record = archive.with_suffix('.manifest.json')
    if archive.exists() and record.exists():
        previous = json.loads(record.read_text())
        if previous['names'] == names and sha(archive) == previous['sha256']:
            print('Reuse:', archive.relative_to(STAGE), flush=True)
            return previous
        raise ValueError('Existing shard differs: %s' % archive)
    directory.mkdir(parents=True, exist_ok=True)
    partial = archive.with_name(archive.name + '.partial')
    entries = []
    with tarfile.open(partial, 'w', format=tarfile.PAX_FORMAT) as tar:
        for index, name in enumerate(names, 1):
            paths = ['images/' + name] + [kind + '/' + str(Path(name).with_suffix('.png'))
                                         for kind in ['mask_sam3', 'semantic_weights']]
            dimensions = []
            for relative in paths:
                source = DATA / relative
                payload = source.read_bytes()
                if not payload:
                    raise ValueError('Empty input: %s' % relative)
                with Image.open(io.BytesIO(payload)) as image:
                    dimensions.append(image.size)
                    image.verify()
                info = tarfile.TarInfo(relative)
                info.size = len(payload)
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(payload))
                entries.append(dict(path=relative, size=info.size, sha256=hashlib.sha256(payload).hexdigest()))
            if len(set(dimensions)) != 1:
                raise ValueError('Annotation dimensions differ: %s' % name)
            if index % 250 == 0:
                print('%s shard %d: %d/%d samples' % (dataset, number, index, len(names)), flush=True)
    partial.replace(archive)
    result = dict(file=archive.name, size=archive.stat().st_size, sha256=sha(archive),
                  names=names, files=entries)
    write_json(record, result)
    print('Built:', archive.relative_to(STAGE), archive.stat().st_size, flush=True)
    return result


def build_data(dataset):
    names = (ROOT / 'experiments/next_16/manifests/images_6650.txt').read_text().splitlines()
    expected = 6650
    if dataset == 'full':
        names = camera_names(DATA / 'sparse/0_full/images.bin')
        expected = 90360
    if len(names) != expected or len(set(names)) != expected:
        raise ValueError('Incorrect dataset size')
    if dataset == 'subset_6650' and set(names) != set(camera_names(DATA / 'sparse/0/images.bin')):
        raise ValueError('Subset camera model differs from training manifest')
    directory = STAGE / 'data' / dataset
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'images.txt').write_text('\n'.join(names) + '\n')
    # Both entry points can use the historical sparse/0_full name independently.
    mappings = {'sparse/0': DATA / 'sparse/0',
                'sparse/0_full': DATA / ('sparse/0_full' if dataset == 'full' else 'sparse/0')}
    metadata = directory / 'metadata.tar'
    with tarfile.open(metadata.with_suffix('.tar.partial'), 'w') as tar:
        for target, source in mappings.items():
            for name in ['cameras.bin', 'images.bin', 'points3D.ply']:
                tar.add(source / name, arcname=target + '/' + name, recursive=False)
    metadata.with_suffix('.tar.partial').replace(metadata)
    groups = [names[i:i + 3000] for i in range(0, len(names), 3000)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(lambda pair: create_shard(dataset, *pair), enumerate(groups)))
    if dataset == 'full':
        full_hashes = {entry['path']: entry['sha256'] for record in records for entry in record['files']}
        for manifest in (STAGE / 'data/subset_6650').glob('samples-*.manifest.json'):
            for entry in json.loads(manifest.read_text())['files']:
                if full_hashes.get(entry['path']) != entry['sha256']:
                    raise ValueError('Subset differs from full data: %s' % entry['path'])
    write_json(directory / 'dataset.json', dict(version='20261014', samples=len(names),
               metadata=dict(file=metadata.name, sha256=sha(metadata)),
               shards=[{k: r[k] for k in ['file', 'size', 'sha256']} for r in records],
               layout='Extract every tar into one data directory; archives are independent.',
               sparse_subdir='sparse/0_full'))


def split_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = Path(str(destination) + '.parts.json')
    if record.exists():
        previous = json.loads(record.read_text())
        if previous['original_size'] == source.stat().st_size and all(
                (destination.parent / p['file']).exists() and sha(destination.parent / p['file']) == p['sha256']
                for p in previous['parts']):
            return
        raise ValueError('Invalid existing file parts')
    whole = hashlib.sha256()
    parts = []
    with source.open('rb') as handle:
        index = 0
        while True:
            first = handle.read(8 * 1024**2)
            if not first:
                break
            part = Path(str(destination) + '.part%03d' % index)
            digest = hashlib.sha256()
            size = 0
            with part.with_name(part.name + '.partial').open('wb') as output:
                chunk = first
                while chunk:
                    output.write(chunk)
                    digest.update(chunk)
                    whole.update(chunk)
                    size += len(chunk)
                    if size >= LIMIT:
                        break
                    chunk = handle.read(min(8 * 1024**2, LIMIT - size))
            part.with_name(part.name + '.partial').replace(part)
            parts.append(dict(file=part.name, size=size, sha256=digest.hexdigest()))
            print('Split:', part.relative_to(STAGE), flush=True)
            index += 1
    expected_file = Path(str(source) + '.sha256')
    if expected_file.exists() and whole.hexdigest() != expected_file.read_text().split()[0]:
        raise ValueError('Original checksum mismatch')
    write_json(record, dict(original=destination.name, original_size=source.stat().st_size,
                           sha256=whole.hexdigest(), parts=parts))


def build_environment():
    source = ROOT / 'packages/fastgs_py310_v1.tar'
    suspicious = []
    with tarfile.open(source, 'r:') as tar:
        for item in tar:
            if Path(item.name).name.lower() in {'.netrc', '.env', '.wandb_auth.json',
                                                'read_token', 'credentials', 'pip.conf'}:
                suspicious.append(item.name)
    write_json(WORK / 'environment_security_check.json', dict(flagged_names=suspicious))
    if suspicious:
        raise ValueError('Review potentially sensitive environment files before upload')
    target = STAGE / 'environment' / ENVIRONMENT / 'industrialgs_py310_v1.tar'
    split_file(source, target)
    manifest = json.loads(Path(str(source) + '.json').read_text())
    if json.loads(Path(str(target) + '.parts.json').read_text())['sha256'] != manifest['sha256']:
        raise ValueError('Environment checksum mismatch')
    manifest['archive'] = target.name
    manifest.pop('source', None)
    write_json(Path(str(target) + '.json'), manifest)


def build_results():
    for source in sorted((ROOT / 'result').rglob('*')):
        if not source.is_file():
            continue
        target = STAGE / source.relative_to(ROOT)
        if source.stat().st_size > LIMIT:
            split_file(source, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)


def api_client():
    from modelscope_hub import HubApi
    return HubApi(token=Path.home().joinpath('.modelscope/read_token').read_text().strip())


def upload(section):
    api = api_client()
    directory = STAGE / section
    api.upload_folder(REPO, 'dataset', directory, path_in_repo=section,
                      max_workers=2, use_cache=True, disable_tqdm=True,
                      tracker_path=WORK / ('tracker_' + section.replace('/', '_') + '.json'),
                      commit_message='Upload verified IndustrialGS ' + section,
                      progress_callback=lambda event: print(json.dumps(event, default=str), flush=True))
    remote = {f.path: f for f in api.list_repo_files(REPO, 'dataset')}
    checked = []
    for path in directory.rglob('*'):
        if not path.is_file():
            continue
        name = path.relative_to(STAGE).as_posix()
        item = remote.get(name)
        if item is None or item.size != path.stat().st_size:
            raise ValueError('Remote size differs: %s' % name)
        digest = sha(path)
        if item.sha256 and item.sha256 != digest:
            raise ValueError('Remote checksum differs: %s' % name)
        checked.append(dict(path=name, size=item.size, sha256=digest,
                            remote_sha256_verified=bool(item.sha256)))
    write_json(WORK / ('verified_' + section.replace('/', '_') + '.json'), checked)
    print('Verified remote section:', section, len(checked), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['build', 'upload'])
    parser.add_argument('section', choices=['data/subset_6650', 'data/full', 'environment', 'result'])
    args = parser.parse_args()
    if args.action == 'upload':
        upload(args.section)
    elif args.section.startswith('data/'):
        build_data(args.section.split('/')[1])
    elif args.section == 'environment':
        build_environment()
    else:
        build_results()


if __name__ == '__main__':
    main()
