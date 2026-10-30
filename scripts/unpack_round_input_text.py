"""Restore the private Library text handoff as one verified ZIP; no execution."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile


def unpack(path,destination,text_sha256,zip_sha256):
    raw=Path(path).read_bytes()
    if len(raw)>100_000_000 or hashlib.sha256(raw).hexdigest()!=text_sha256:
        raise ValueError('Input text size/SHA differs')
    first,body=raw.decode().split('\nBEGIN_BASE64\n',1)
    encoded,tail=body.split('\nEND_BASE64',1)
    if tail.strip():raise ValueError('Unexpected trailing data')
    header=json.loads(first)
    if header.get('format')!='shogi-round-inputs-base64zip' or header.get('round_id') not in (3,4,5):
        raise ValueError('Unexpected input format or round')
    payload=base64.b64decode(''.join(encoded.splitlines()),validate=True)
    digest=hashlib.sha256(payload).hexdigest()
    if len(payload)!=header['zip_bytes'] or digest!=header['zip_sha256'] or digest!=zip_sha256:
        raise ValueError('Input ZIP size/SHA differs')
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        allowed={'data.npz','data-manifest.json','start-policy.json','SHA256SUMS.json'}
        if len(z.namelist())!=4 or set(z.namelist())!=allowed:raise ValueError('Unexpected ZIP members')
        checks=json.loads(z.read('SHA256SUMS.json'))
        if set(checks)!=allowed-{'SHA256SUMS.json'}:raise ValueError('Incomplete checksum list')
        for name,expected in checks.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=expected:raise ValueError('Input item checksum differs')
        manifest=json.loads(z.read('data-manifest.json'))
        if manifest['round_id']!=header['round_id'] or manifest['warm_start_file_sha256']!=checks['start-policy.json']:
            raise ValueError('Round/model identity differs')
    destination=Path(destination)
    if destination.exists() and destination.read_bytes()!=payload:raise ValueError('Refusing to overwrite another input')
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(payload)
    return {'path':str(destination.resolve()),'round_id':header['round_id'],'bytes':len(payload),'sha256':digest}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('path',type=Path);p.add_argument('destination',type=Path)
    p.add_argument('--text-sha256',required=True);p.add_argument('--zip-sha256',required=True)
    a=p.parse_args();print(json.dumps(unpack(a.path,a.destination,a.text_sha256,a.zip_sha256),indent=2))
