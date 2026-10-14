"""Bounded background PNG uploads; a completed future means both outputs exist."""
import base64
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import io
import os
from pathlib import Path
import shlex
import threading
import time
import uuid

from PIL import Image


class LocalStore:
    def __init__(self, root):
        self.root = Path(root)
        self.parents = set()
        self.lock = threading.Lock()

    def existing(self, folder):
        base = self.root / folder
        return {p.relative_to(base).as_posix() for p in base.rglob('*.png')}

    def put(self, relative, payload):
        target = self.root / relative
        with self.lock:
            if target.parent not in self.parents:
                target.parent.mkdir(parents=True, exist_ok=True)
                self.parents.add(target.parent)
        temp = target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            temp.write_bytes(payload)
            try:
                os.link(temp, target)  # Atomic publication, never replace an existing PNG.
            except FileExistsError:
                return False
            return True
        finally:
            temp.unlink(missing_ok=True)

    def status(self, path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.tmp')
        temp.write_bytes(payload)
        os.replace(temp, path)


class OssStore:
    def __init__(self, root, credentials, mount='/data/oss_bucket_0'):
        import oss2
        self.oss2 = oss2
        self.mount = Path(mount)
        self.prefix = self.key(Path(root)).rstrip('/')
        values = {}
        # Read literal assignments only; do not execute a credentials shell script.
        for line in Path(credentials).read_text().splitlines():
            for word in shlex.split(line, comments=True):
                if '=' in word:
                    name, value = word.split('=', 1)
                    if name in ('FMC_AK', 'FMC_SK', 'FMC_ENDPOINT', 'FMC_BUCKET'):
                        values[name] = value
        for name in ('FMC_AK', 'FMC_SK', 'FMC_ENDPOINT'):
            if not values.get(name):
                raise ValueError(f'Missing credential field: {name}')
        self.bucket_name = values.get('FMC_BUCKET', 'fmc')
        if self.bucket_name != 'fmc':
            raise ValueError('Expected fmc bucket for /data/oss_bucket_0')
        self.endpoint = 'https://' + values['FMC_ENDPOINT'].removeprefix('https://').removeprefix('http://').rstrip('/')
        self.auth = oss2.Auth(values['FMC_AK'], values['FMC_SK'])
        self.local = threading.local()

    def key(self, path):
        return Path(os.path.abspath(path)).relative_to(self.mount).as_posix()

    def bucket(self):
        if not hasattr(self.local, 'bucket'):
            self.local.bucket = self.oss2.Bucket(self.auth, self.endpoint, self.bucket_name,
                                               connect_timeout=30, enable_crc=True)
        return self.local.bucket

    def existing(self, folder):
        prefix = f'{self.prefix}/{folder}/'
        return {obj.key[len(prefix):] for obj in self.oss2.ObjectIterator(self.bucket(), prefix=prefix)
                if obj.key.endswith('.png')}

    def _put(self, key, payload, overwrite=False):
        headers = {'Content-MD5': base64.b64encode(hashlib.md5(payload).digest()).decode(),
                   'Content-Type': 'application/json' if overwrite else 'image/png'}
        if not overwrite:
            headers['x-oss-forbid-overwrite'] = 'true'
        for attempt in range(4):
            try:
                self.bucket().put_object(key, payload, headers=headers)
                return True
            except self.oss2.exceptions.ServerError as exc:
                if not overwrite and exc.status == 409 and exc.code in ('FileAlreadyExists', 'ObjectAlreadyExists'):
                    return False
                if attempt == 3 or (exc.status < 500 and exc.status != 429):
                    raise RuntimeError(f'OSS upload failed: key={key}, status={exc.status}, code={exc.code}') from None
            except self.oss2.exceptions.RequestError:
                if attempt == 3:
                    raise RuntimeError(f'OSS network upload failed: key={key}') from None
            time.sleep(2 ** attempt)

    def put(self, relative, payload):
        return self._put(f'{self.prefix}/{relative}', payload)

    def status(self, path, payload):
        self._put(self.key(path), payload, overwrite=True)


class ImageWriter:
    def __init__(self, store, workers=8, capacity=64):
        if workers < 1 or capacity < 1:
            raise ValueError('Upload workers and capacity must be positive')
        self.store = store
        self.capacity = capacity
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='png-upload')
        self.futures = set()

    def _save(self, name, outputs):
        started = time.monotonic()
        encode_seconds = upload_seconds = 0
        for folder, array in outputs.items():
            start = time.monotonic()
            encoded = io.BytesIO()
            Image.fromarray(array).save(encoded, format='PNG')
            payload = encoded.getvalue()
            with Image.open(io.BytesIO(payload)) as image:
                image.verify()
            encode_seconds += time.monotonic() - start
            start = time.monotonic()
            self.store.put(f'{folder}/{Path(name).with_suffix(".png").as_posix()}', payload)
            upload_seconds += time.monotonic() - start
        return dict(image=name, encode_seconds=encode_seconds, upload_seconds=upload_seconds,
                    worker_seconds=time.monotonic() - started)

    def submit(self, name, outputs):
        if len(self.futures) >= self.capacity:
            raise RuntimeError('Upload queue full; drain before submitting')
        self.futures.add(self.executor.submit(self._save, name, outputs))

    def collect(self, timeout=0):
        if not self.futures:
            return []
        ready, _ = wait(self.futures, timeout=timeout, return_when=FIRST_COMPLETED)
        results = []
        for future in ready:
            results.append(future.result())  # Propagate failure before reporting complete.
            self.futures.remove(future)
        return results

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
