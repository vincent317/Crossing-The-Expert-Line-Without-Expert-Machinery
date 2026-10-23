"""Restore fingerprinted historical sources and apply the FP32 baseline repairs."""
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile


def restore(destination):
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'repair.json').read_text())
    entry = (root / '../baseline-retest-20261023/baseline-topk-r2-entry.sh').read_text()
    archives = []
    for blob in re.findall(r'^printf %s (\S+) \| base64 -d \| tar x\w*', entry, re.M):
        archive = tarfile.open(fileobj=io.BytesIO(base64.b64decode(blob)))
        if 'historical-topk/ref_ako_fs6/kernel.py' in archive.getnames():
            archives.append(archive)
        else:
            archive.close()
    if len(archives) != 1:
        raise ValueError('Expected exactly one historical baseline archive')
    destination.mkdir(parents=True, exist_ok=False)
    with archives[0] as archive:
        for name, expected in manifest['source_sha256'].items():
            path = Path(name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError(f'Invalid source path: {name}')
            data = archive.extractfile('historical-topk/' + name).read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f'Historical source hash mismatch: {name}')
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    for name in manifest['patches']:
        patch = str(root / name)
        subprocess.run(['git', 'apply', '--check', patch], cwd=destination, check=True)
        subprocess.run(['git', 'apply', patch], cwd=destination, check=True)
    for name, expected in manifest['repaired_sha256'].items():
        if hashlib.sha256((destination / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Repaired source hash mismatch: {name}')
    print('REPAIRED_SOURCE_VERIFIED', destination)


if __name__ == '__main__':
    restore(Path(sys.argv[1]).resolve())
