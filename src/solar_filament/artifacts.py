import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(f"Refusing to overwrite artifact: {path}")
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".pending")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            # Publish exclusively on both Windows and Linux, including concurrent writers.
            os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def create_run_directory(root: Path, name: str) -> Path:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name) is None:
        raise ValueError("Run names must contain 1-64 letters, digits, underscores or hyphens")
    path = root / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def source_identity(package: Path | None = None) -> str:
    package = package or Path(__file__).parent
    return object_sha256({p.name: file_sha256(p) for p in sorted(package.glob("*.py"))})
