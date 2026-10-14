"""Read-only OSS coverage/PNG sample audit and COLMAP coordinate comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import struct
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'deployment/sam3_full'))
from output_store import OssStore


def poses(path):
    output = {}
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            values = struct.unpack('<idddddddi', f.read(64))
            name = bytearray()
            while True:
                c = f.read(1)
                if not c:
                    raise ValueError('truncated COLMAP image name')
                if c == b'\0':
                    break
                name.extend(c)
            points = struct.unpack('<Q', f.read(8))[0]
            f.seek(points*24, 1)
            output[name.decode()] = values
    assert len(output) == n
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--credentials', required=True)
    ap.add_argument('--status-dir', required=True)
    ap.add_argument('--report', required=True)
    ap.add_argument('--samples', type=int, default=256)
    args = ap.parse_args()
    data = Path(args.data)
    store = OssStore(data, args.credentials)
    small, full = poses(data/'sparse/0/images.bin'), poses(data/'sparse/0_full/images.bin')
    names = sorted(full)
    missing_small = sorted(set(small)-set(full))
    matching = [n for n in small if n in full]
    q_delta = max(min(np.max(np.abs(np.array(small[n][1:5])-full[n][1:5])),
                      np.max(np.abs(np.array(small[n][1:5])+full[n][1:5]))) for n in matching)
    t_delta = max(float(np.max(np.abs(np.array(small[n][5:8])-full[n][5:8]))) for n in matching)
    cameras_identical = (data/'sparse/0/cameras.bin').read_bytes() == (data/'sparse/0_full/cameras.bin').read_bytes()
    point_hashes = {p: hashlib.sha256((data/p).read_bytes()).hexdigest()
                    for p in ('sparse/0/points3D.ply', 'sparse/0_full/points3D.ply')}

    def inventory(folder):
        prefix = f'{store.prefix}/{folder}/'
        objects = {o.key[len(prefix):]: o.size for o in
                   store.oss2.ObjectIteratorV2(store.bucket(), prefix=prefix, max_keys=1000)}
        expected = set(names if folder == 'images' else [str(Path(n).with_suffix('.png')) for n in names])
        present = {k for k in objects if k.lower().endswith(('.jpg', '.jpeg', '.png'))}
        missing = sorted(expected-present)
        empty = sorted(k for k in expected & present if not objects[k])
        result = dict(expected=len(expected), present=len(present), missing_count=len(missing),
                      missing_first=missing[:20], zero_bytes_count=len(empty), zero_bytes_first=empty[:20],
                      extra_count=len(present-expected), bytes=sum(objects[k] for k in expected & present))
        print(folder, json.dumps(result), flush=True)
        return folder, result

    with ThreadPoolExecutor(max_workers=3) as pool:
        inventories = dict(pool.map(inventory, ['images', 'mask_sam3', 'semantic_weights']))
    states = [json.loads(store.bucket().get_object(store.key(Path(args.status_dir)/f'shard_{i}.json')).read())
              for i in range(8)]
    selected = [names[i*len(names)//args.samples] for i in range(min(args.samples, len(names)))]

    def verify(name):
        with Image.open(io.BytesIO(store.bucket().get_object(store.prefix+'/images/'+name).read())) as image:
            image.load()
            size = image.size
        fractions = {}
        for folder, labels in [('mask_sam3', {0,255}), ('semantic_weights', {0,100,200})]:
            key = store.prefix+'/'+folder+'/'+str(Path(name).with_suffix('.png'))
            payload = store.bucket().get_object(key).read()
            with Image.open(io.BytesIO(payload)) as im:
                im.verify()
            with Image.open(io.BytesIO(payload)) as im:
                array = np.array(im)
                assert im.size == size, (name, folder, im.size, size)
                assert array.ndim == 2 and set(np.unique(array)) <= labels, (name, folder)
                fractions[folder] = float(np.mean(array == (255 if folder == 'mask_sam3' else 0)))
        return dict(name=name, width=size[0], height=size[1], **fractions)

    with ThreadPoolExecutor(max_workers=12) as pool:
        samples = list(pool.map(verify, selected))
    report = dict(checked_at=datetime.now(timezone.utc).isoformat(), source=str(data),
                  inventory=inventories, sample_count=len(samples), samples=samples,
                  png_validation='full name/size inventory; stratified PNG decode/CRC/labels/dimensions sample, not full-file decode',
                  pose_audit=dict(subset=len(small), full=len(full), missing_subset=missing_small,
                                  max_quaternion_abs_delta=float(q_delta), max_translation_abs_delta=t_delta,
                                  camera_ids_match=all(small[n][-1] == full[n][-1] for n in matching),
                                  cameras_bin_identical=cameras_identical, initial_ply_sha256=point_hashes),
                  shard_states=states,
                  coverage_complete=all(v['missing_count']==v['zero_bytes_count']==0 for v in inventories.values()),
                  all_shards_complete=all(s['state']=='complete' and s['completed']==s['pending'] and
                                          s['upload_queue']==0 for s in states))
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('samples','shard_states')}), flush=True)


if __name__ == '__main__':
    main()
