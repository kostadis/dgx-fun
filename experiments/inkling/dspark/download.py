"""Download pinned Unsloth ARM64 runtime and text-only Mini/DSpark assets."""
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time
import urllib.error
import urllib.request


ROOT = Path('/work')
REPO = 'unsloth/Inkling-2-Mini-GGUF'
REVISION = 'ae9327c5fbc7333116ba32eee78bb5d2bbefae4f'
TAG = 'b12588-mix-c0c3de8'
ARCHIVE = 'app-b12588-mix-c0c3de8-linux-arm64-cuda13-portable.tar.gz'
BINARY_SHA = '1ae89924180ec0ce8a18714737e591bf171db70f4fc5f5bd9fc160c7455004c7'
FILES = {
    'Inkling-2-Mini-UD-Q4_K_XL.gguf': 'e62d463c135ffe283df8d94b279a96ca145ac128b52ce854478613045bf4de15',
    'dspark-Inkling-2-Mini-Q8_0.gguf': '2ec78510da837e8fc3be54a14757a0c5fbd2d8d9e85a0d92853fafdf674f3773',
}
SIZES = {'Inkling-2-Mini-UD-Q4_K_XL.gguf': 17629759456,
         'dspark-Inkling-2-Mini-Q8_0.gguf': 402278560}


def download(url, path, expected_size=None):
    """Stream at most a few MiB at once and resume interrupted transfers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (expected_size is None or path.stat().st_size == expected_size):
        return
    partial = path.with_name(path.name + '.part')
    for attempt in range(6):
        offset = partial.stat().st_size if partial.exists() else 0
        if expected_size is not None and offset == expected_size:
            partial.replace(path)
            return
        headers = {'User-Agent': 'dgx-fun-inkling-dspark-trial', 'Accept-Encoding': 'identity'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as source:
                resumed = source.status == 206
                if resumed:
                    assert source.headers['Content-Range'].startswith(f'bytes {offset}-')
                else:
                    offset = 0
                updated = 0
                with partial.open('ab' if resumed else 'wb') as target:
                    while block := source.read(1024**2):
                        target.write(block)
                        offset += len(block)
                        if time.monotonic() - updated > 5:
                            progress = {'file': path.name, 'bytes': offset,
                                        'expected_bytes': expected_size, 'attempt': attempt}
                            (ROOT / 'download-progress.json').write_text(json.dumps(progress) + '\n')
                            updated = time.monotonic()
            if expected_size is not None:
                assert partial.stat().st_size == expected_size
            partial.replace(path)
            return
        except (OSError, urllib.error.URLError):
            if attempt == 5:
                raise
            time.sleep(3)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def main():
    started = time.time()
    path = ROOT / ARCHIVE
    download(f'https://github.com/unslothai/llama.cpp/releases/download/{TAG}/{ARCHIVE}', path, 252263504)
    assert digest(path) == BINARY_SHA, 'Runtime archive checksum differs'
    directory = ROOT / 'runtime'
    directory.mkdir(exist_ok=True)
    with tarfile.open(path) as archive:
        archive.extractall(directory, filter='data')
    print('Pinned ARM64 CUDA 13 runtime verified and extracted.', flush=True)
    transport = os.environ.get('INKLING_DOWNLOAD_TRANSPORT', 'http')
    if transport == 'xet':
        from huggingface_hub import snapshot_download
        snapshot_download(REPO, revision=REVISION, token=False, local_dir=ROOT/'models',
                          allow_patterns=[*FILES, 'dspark/README.md'], max_workers=1)
    else:
        assert transport == 'http'
        for name in FILES:
            download(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}?download=true',
                     ROOT / 'models' / name, SIZES[name])
        download(f'https://huggingface.co/{REPO}/resolve/{REVISION}/dspark/README.md',
                 ROOT / 'models/dspark/README.md', 3113)
    verified = {}
    for name, expected in FILES.items():
        model = ROOT / 'models' / name
        actual = digest(model)
        assert actual == expected, f'{name} checksum differs'
        verified[name] = {'sha256': actual, 'bytes': model.stat().st_size}
        print('Verified', name, model.stat().st_size, flush=True)
    result = {'passed': True, 'repo': REPO, 'revision': REVISION, 'files': verified,
        'runtime': {'release': TAG, 'archive': ARCHIVE, 'sha256': BINARY_SHA},
        'started_unix': started, 'finished_unix': time.time(), 'transport': transport,
        'vision_projector_downloaded': False}
    (ROOT / 'download-receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print('All DSpark trial assets ready.', flush=True)


if __name__ == '__main__':
    main()
