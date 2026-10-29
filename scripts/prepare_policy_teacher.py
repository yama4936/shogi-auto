"""Build a pinned official GPL CPU engine; extract only reviewed NNUE data.

Downloads public source and the official tanuki- weights-only release. No
precompiled executable, opening book, private record or API key is used.
"""
import argparse
import ctypes as c
import ctypes.util
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import time
import urllib.request

REPOSITORY = 'https://github.com/yaneurao/YaneuraOu.git'
TAG = 'v8.60git'
COMMIT = 'eb2856f91e6088e5b8c1216e3d84d0563f7cd85f'
WEIGHTS_URL = ('https://github.com/nodchip/tanuki-/releases/download/'
               'tanuki-.halfkp_256x2-32-32.2023-05-08/tanuki-.halfkp_256x2-32-32.2023-05-08.7z')
ARCHIVE_SHA = 'f16dc66c529857bf08fce8cee4f1e53a6993297644100267a6ccccd62e43dbba'
WEIGHTS_SHA = '1141d275bceec911156801f27303dc9ff5beb24f4f59144cc069306c59e80782'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract(archive, destination):
    """Read known paths through system libarchive; never extract executables."""
    name = ctypes.util.find_library('archive')
    if not name:raise RuntimeError('System libarchive is required for the official 7z asset')
    lib = c.CDLL(name)
    lib.archive_read_new.restype=c.c_void_p
    for function in ('archive_read_support_filter_all','archive_read_support_format_all','archive_read_free'):
        getattr(lib,function).argtypes=[c.c_void_p]
    lib.archive_read_open_filename.argtypes=[c.c_void_p,c.c_char_p,c.c_size_t]
    lib.archive_read_next_header.argtypes=[c.c_void_p,c.POINTER(c.c_void_p)]
    lib.archive_entry_pathname.argtypes=[c.c_void_p];lib.archive_entry_pathname.restype=c.c_char_p
    lib.archive_entry_size.argtypes=[c.c_void_p];lib.archive_entry_size.restype=c.c_longlong
    lib.archive_read_data.argtypes=[c.c_void_p,c.c_void_p,c.c_size_t];lib.archive_read_data.restype=c.c_ssize_t
    pointer=lib.archive_read_new();lib.archive_read_support_filter_all(pointer);lib.archive_read_support_format_all(pointer)
    if lib.archive_read_open_filename(pointer,str(archive).encode(),10240)!=0:raise ValueError('Cannot open verified weights archive')
    entry=c.c_void_p();seen=set()
    try:
        while True:
            status=lib.archive_read_next_header(pointer,c.byref(entry))
            if status==1:break
            if status!=0:raise ValueError('Invalid weights archive')
            path=lib.archive_entry_pathname(entry).decode('utf-8')
            if path in seen or path not in ('eval/','eval/nn.bin','gpl-3.0.txt'):
                raise ValueError(f'Unexpected path in pinned weights archive: {path}')
            seen.add(path)
            if path=='eval/':continue
            size=lib.archive_entry_size(entry)
            if size != (64217066 if path=='eval/nn.bin' else 35823):raise ValueError('Unexpected archived file size')
            target=destination/PurePosixPath(path);target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_suffix('.tmp');total=0
            with temporary.open('wb') as out:
                buffer=c.create_string_buffer(1024*1024)
                while True:
                    n=lib.archive_read_data(pointer,buffer,len(buffer))
                    if n==0:break
                    if n<0:raise ValueError('Corrupt weights archive data')
                    total+=n
                    if total>size:raise ValueError('Excessive weights archive data')
                    out.write(buffer.raw[:n])
            if total!=size:raise ValueError('Truncated weights archive data')
            temporary.replace(target)
    finally:lib.archive_read_free(pointer)
    if seen!={'eval/','eval/nn.bin','gpl-3.0.txt'} or sha(destination/'eval/nn.bin')!=WEIGHTS_SHA:
        raise ValueError('Extracted teacher weights checksum mismatch')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    a=p.parse_args();root=a.directory.resolve();root.mkdir(parents=True,exist_ok=True)
    for command in ('git','make','g++'):
        if not shutil.which(command):raise RuntimeError(f'Existing build tool required: {command}')
    started=time.monotonic();source=root/'YaneuraOu-source';asset=root/'tanuki-2023-05-08.7z';teacher=root/'teacher'
    if not source.exists():
        subprocess.run(['git','clone','--depth','1','--branch',TAG,REPOSITORY,str(source)],check=True,timeout=120)
    actual=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if actual!=COMMIT:raise ValueError('Teacher source commit mismatch')
    subprocess.run(['git','-C',str(source),'diff','--exit-code','HEAD'],check=True,capture_output=True)
    if not asset.exists():
        size=0;temporary=asset.with_suffix('.part')
        with urllib.request.urlopen(WEIGHTS_URL,timeout=30) as response,temporary.open('wb') as out:
            while data:=response.read(1024*1024):
                size+=len(data)
                if size>23966109:raise ValueError('Unexpected release asset size')
                out.write(data)
        if size!=23966109 or sha(temporary)!=ARCHIVE_SHA:raise ValueError('Official release asset checksum mismatch')
        temporary.replace(asset)
    if sha(asset)!=ARCHIVE_SHA:raise ValueError('Cached release asset checksum mismatch')
    extract(asset,teacher)
    build=['make','-j2','normal','COMPILER=g++','TARGET_CPU=SSE42','YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE']
    with (root/'build.log').open('w') as log:
        subprocess.run(build,cwd=source/'source',stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    binary=source/'source/YaneuraOu-by-gcc'
    manifest={'repository':REPOSITORY,'tag':TAG,'commit':COMMIT,'build_command':build,
              'source_license':'GPL-3.0; upstream Copying.txt','weights_url':WEIGHTS_URL,
              'weights_archive_sha256':ARCHIVE_SHA,'weights_sha256':WEIGHTS_SHA,
              'weights_license':'GPL-3.0; gpl-3.0.txt in the official weights-only archive',
              'binary':str(binary),'binary_sha256':sha(binary),'eval_dir':str(teacher/'eval'),
              'source_changed':False,'book_used':False,'precompiled_binary_used':False,
              'seconds':time.monotonic()-started}
    (root/'teacher-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
