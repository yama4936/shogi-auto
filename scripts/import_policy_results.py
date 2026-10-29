"""Verify and unpack the Notebook's text-only CPU handoff; never load pickle."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

from shogi_auto.policy.network import load_network


def import_results(path, destination, expected_text_sha256):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_text_sha256:
        raise ValueError('Received text-file SHA256 differs from the parent handoff')
    text=raw.decode('utf-8')
    first,body=text.split('\nBEGIN_BASE64\n',1)
    encoded,tail=body.split('\nEND_BASE64',1)
    if tail.strip():raise ValueError('Unexpected data after result block')
    header=json.loads(first)
    if header.get('format')!='shogi-policy-v2-results-base64zip':raise ValueError('Unsupported result format')
    payload=base64.b64decode(''.join(encoded.splitlines()),validate=True)
    if len(payload)>20_000_000 or len(payload)!=header['zip_bytes'] or hashlib.sha256(payload).hexdigest()!=header['zip_sha256']:
        raise ValueError('Result ZIP size or SHA256 differs')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    allowed={'initial-policy.json','best-policy.json','training-report.json','training-settings.json',
             'run-report.json','optimizer-accounting.json','training-process.log','SHA256SUMS.json'}
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or not set(names)<=allowed or sum(v.file_size for v in z.infolist())>50_000_000:
            raise ValueError('Unexpected result archive contents')
        checks=json.loads(z.read('SHA256SUMS.json'))
        if set(checks)!=set(names)-{'SHA256SUMS.json'}:raise ValueError('Incomplete result checksums')
        for name,expected in checks.items():
            data=z.read(name)
            if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Result item checksum mismatch')
            target=destination/name
            if target.exists() and target.read_bytes()!=data:raise ValueError('Refusing to overwrite a different experiment')
            if name.endswith('.json'):json.loads(data)
        for name in names:(destination/name).write_bytes(z.read(name))
    for name in ('initial-policy.json','best-policy.json'):
        if (destination/name).exists():load_network(destination/name)
    if (destination/'training-report.json').exists():
        report=json.loads((destination/'training-report.json').read_text())
        if report['source']['commit']!=header['source_commit']:raise ValueError('Training/source handoff mismatch')
        manifest=json.loads((destination/'run-report.json').read_text())
        if manifest['data_zip_sha256']!=header['data_zip_sha256']:raise ValueError('Training/data handoff mismatch')
    return header


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('path',type=Path)
    p.add_argument('--destination',type=Path,required=True);p.add_argument('--text-sha256',required=True)
    a=p.parse_args();print(json.dumps(import_results(a.path,a.destination,a.text_sha256),indent=2))
