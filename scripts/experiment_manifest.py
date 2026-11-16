"""Bind external artifacts to an exact clean source commit, without storing paths."""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import re
import subprocess

def git(root, *args):
    return subprocess.check_output(["git", "--no-optional-locks", "-C", str(root), *args], text=True).strip()

def artifacts(items):
    result = {}
    for item in items:
        label, sep, name = item.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", label) or label in result:
            raise ValueError("invalid or duplicate artifact label")
        path = pathlib.Path(name)
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as f:
            before = os.fstat(f.fileno())
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
            after = os.fstat(f.fileno())
        current = path.stat()
        identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        if identity(before) != identity(after) or identity(after) != identity(current) or size != after.st_size:
            raise ValueError("artifact changed while hashing")
        result[label] = {"sha256": digest.hexdigest(), "bytes": size}
    if not result:
        raise ValueError("at least one artifact required")
    return result

def record(root, run_id, output, items):
    root = pathlib.Path(git(root, "rev-parse", "--show-toplevel")).resolve()
    output = pathlib.Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("manifest must be stored outside the source checkout")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise ValueError("invalid experiment ID")
    before = git(root, "rev-parse", "HEAD")
    if git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("source checkout is dirty")
    payload = {"schema": 1, "id": run_id, "code_commit": before,
               "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "artifacts": artifacts(items)}
    if git(root, "rev-parse", "HEAD") != before or git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("source changed during recording")
    # Exclusive create: preserve every existing receipt. Parent must already exist.
    with output.open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return payload

def verify(root, manifest, items):
    payload = json.loads(pathlib.Path(manifest).read_text())
    if payload["schema"] != 1 or not re.fullmatch(r"[0-9a-f]{40,64}", payload["code_commit"]):
        raise ValueError("unsupported or malformed manifest")
    if git(root, "rev-parse", "HEAD") != payload["code_commit"]:
        raise ValueError("source commit mismatch")
    if git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("source checkout is dirty")
    if artifacts(items) != payload["artifacts"]:
        raise ValueError("artifact labels, sizes or hashes differ")
    if git(root, "rev-parse", "HEAD") != payload["code_commit"] or git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("source changed during verification")
    return payload

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("record")
    a.add_argument("--id", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--artifact", action="append", required=True)
    b = sub.add_parser("verify")
    b.add_argument("manifest")
    b.add_argument("--artifact", action="append", required=True)
    args = p.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    try:
        if args.command == "record":
            record(root, args.id, args.output, args.artifact)
        else:
            verify(root, args.manifest, args.artifact)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        p.exit(1, type(exc).__name__ + ": validation failed; no execution authorized\n")
    print("manifest " + args.command + " passed")

if __name__ == "__main__":
    main()
