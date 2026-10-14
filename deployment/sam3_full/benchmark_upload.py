"""Compare old mounted writes and direct SDK uploads in an isolated OSS prefix."""
import argparse
import io
import json
from pathlib import Path
import time
import uuid
import os

import numpy as np
from PIL import Image
from output_store import OssStore, ImageWriter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--credentials', required=True)
    ap.add_argument('--samples', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--report', required=True)
    args = ap.parse_args()
    root = Path(args.root)
    images = []
    for p in sorted((Path(args.samples)/'mask_sam3').rglob('*.png'))[:8]:
        relative = p.relative_to(Path(args.samples)/'mask_sam3')
        with Image.open(p) as im:
            mask = np.array(im)
        with Image.open(Path(args.samples)/'semantic_weights'/relative) as im:
            weights = np.array(im)
        images.append((relative.with_suffix('.jpg').as_posix(), {'mask_sam3': mask, 'semantic_weights': weights}))
    assert len(images) == 8
    store = OssStore(root/'sdk', args.credentials)
    writer = ImageWriter(store, workers=8, capacity=64)
    start = time.monotonic()
    completed = []
    try:
        for name, outputs in images:
            writer.submit(name, outputs)
        while writer.futures:
            completed.extend(writer.collect(timeout=1))
            if time.monotonic()-start > 30:
                print(f'SDK completed={len(completed)}/8', flush=True)
    finally:
        writer.close()
    sdk_seconds = time.monotonic()-start
    print(f'SDK: 8 images / 16 PNGs in {sdk_seconds:.3f}s', flush=True)
    name = images[0][0]
    key = f'mask_sam3/{Path(name).with_suffix(".png").as_posix()}'
    full_key = f'{store.prefix}/{key}'
    old = store.bucket().get_object(full_key).read()
    assert store.put(key, b'do not overwrite existing PNG') is False
    assert store.bucket().get_object(full_key).read() == old
    for name, outputs in images:
        for folder, array in outputs.items():
            key = f'{store.prefix}/{folder}/{Path(name).with_suffix(".png").as_posix()}'
            with Image.open(io.BytesIO(store.bucket().get_object(key).read())) as im:
                assert np.array_equal(array, np.array(im))
    print('SDK readback and no-overwrite passed', flush=True)
    assert len(store.existing('mask_sam3')) == 8
    start = time.monotonic()
    for i, (name, outputs) in enumerate(images):
        for folder, array in outputs.items():
            path = root/'mount'/folder/Path(name).with_suffix('.png')
            path.parent.mkdir(parents=True, exist_ok=True)
            assert not path.exists()
            temp = path.with_name(path.stem+'.'+uuid.uuid4().hex+'.tmp')
            try:
                encoded = io.BytesIO()
                Image.fromarray(array).save(encoded, format='PNG')
                encoded.seek(0)
                with Image.open(encoded) as im:
                    im.verify()
                temp.write_bytes(encoded.getvalue())
                os.replace(temp, path)
            finally:
                if temp.exists():
                    temp.unlink()
        print(f'Mount: saved {i+1}/8 images, elapsed={time.monotonic()-start:.3f}s', flush=True)
    mount_seconds = time.monotonic()-start
    result = dict(root=str(root), images=8, sdk_seconds=sdk_seconds, mount_seconds=mount_seconds,
                  sdk_images_per_second=8/sdk_seconds, mount_images_per_second=8/mount_seconds,
                  no_overwrite_verified=True, decoded_pixels_verified=True)
    Path(args.report).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
