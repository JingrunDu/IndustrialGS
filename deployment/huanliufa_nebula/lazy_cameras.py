"""Bounded CPU preparation, main-thread CUDA construction, local compressed cache."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import os
from pathlib import Path
import shutil
import tempfile


class CameraCache:
    def __init__(self, infos, scale, args, prepare, construct):
        self.infos, self.scale, self.args = infos, scale, args
        self.prepare, self.construct = prepare, construct
        self.limit = int(float(os.environ.get('CAMERA_CACHE_GIB', '8')) * 1024**3)
        self.cache, self.pending = OrderedDict(), {}
        self.bytes = 0
        self.workers = int(os.environ.get('CAMERA_LOAD_WORKERS', '8'))
        self.pool = ThreadPoolExecutor(max_workers=self.workers)
        identity = hashlib.sha256(args.source_path.encode()).hexdigest()[:16]
        self.disk = Path(os.environ.get('CAMERA_DISK_CACHE', '/tmp/fastgs_images')) / identity

    def prepare_one(self, index):
        info = self.infos[index]
        rel = Path(info.image_path).relative_to(Path(self.args.source_path) / 'images')
        paths = [Path('images') / rel, Path('mask_sam3') / rel.with_suffix('.png')]
        if self.args.use_semantic_weight:
            paths.append(Path('semantic_weights') / rel.with_suffix('.png'))
        for relative in paths:
            dest = self.disk / relative
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=dest.parent)
                os.close(fd)
                try:
                    shutil.copyfile(Path(self.args.source_path) / relative, tmp)
                    os.replace(tmp, dest)
                finally:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
        args = copy.copy(self.args)
        args.source_path = str(self.disk)
        info = info._replace(image=None, image_path=str(self.disk / 'images' / rel))
        return self.prepare(args, index, info, self.scale)

    def prefetch(self, indices):
        for index in indices:
            if len(self.pending) >= self.workers * 2:
                break
            if index not in self.cache and index not in self.pending:
                self.pending[index] = self.pool.submit(self.prepare_one, index)

    def get(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index][0]
        future = self.pending.pop(index, None)
        kwargs = future.result() if future else self.prepare_one(index)
        camera = self.construct(**kwargs)
        size = sum(t.numel() * t.element_size() for t in
                   (camera.original_image, camera.gt_alpha_mask, camera.semantic_weight) if t is not None)
        while self.cache and self.bytes + size > self.limit:
            _, (_, removed) = self.cache.popitem(last=False)
            self.bytes -= removed
        # One oversized camera remains usable but is never retained in the cache.
        if size <= self.limit:
            self.cache[index] = (camera, size)
            self.bytes += size
        return camera


class LazyCamera:
    def __init__(self, cache, index):
        self.cache, self.index = cache, index

    def materialize(self):
        return self.cache.get(self.index)

    def __getattr__(self, name):
        return getattr(self.materialize(), name)


def lazy_camera_list(infos, scale, args, prepare, construct):
    if not infos:
        return []
    cache = CameraCache(infos, scale, args, prepare, construct)
    print(f'[相机] {len(infos)} 个按需加载，解码缓存上限 {cache.limit / 1024**3:.1f} GiB，'
          f'本地图片缓存 {cache.disk}', flush=True)
    return [LazyCamera(cache, i) for i in range(len(infos))]
