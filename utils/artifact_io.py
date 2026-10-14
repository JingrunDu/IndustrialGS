"""Serialize locally, then publish a closed file and its checksum to the output mount."""
import hashlib
import os
from pathlib import Path
import shutil
import tempfile


def publish_file(local, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    temporary = destination.with_name(destination.name + '.partial')
    with open(local, 'rb') as source, open(temporary, 'wb') as target:
        while True:
            chunk = source.read(8 * 1024**2)
            if not chunk:
                break
            digest.update(chunk)
            target.write(chunk)
    os.replace(temporary, destination)
    marker = destination.with_name(destination.name + '.sha256')
    pending_marker = marker.with_suffix('.partial')
    pending_marker.write_text(digest.hexdigest() + '\n')
    os.replace(pending_marker, marker)


def local_checkpoint(path, directory):
    path = Path(path)
    expected = Path(str(path) + '.sha256').read_text().strip()
    local = Path(directory) / path.name
    digest = hashlib.sha256()
    with path.open('rb') as source, local.open('wb') as target:
        while True:
            chunk = source.read(8 * 1024**2)
            if not chunk:
                break
            digest.update(chunk)
            target.write(chunk)
    if digest.hexdigest() != expected:
        raise ValueError('Checkpoint checksum mismatch; incomplete or corrupted upload')
    return local
