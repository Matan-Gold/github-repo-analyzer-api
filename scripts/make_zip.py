"""Create submission.zip from repository root while excluding local/dev artifacts."""

from __future__ import annotations

from pathlib import Path
import zipfile


EXCLUDE_DIRS = {
    ".git",
    "venv",
    ".venv",
    ".venv.py39.bak",
    "__pycache__",
    ".pytest_cache",
    ".idea",
    ".vscode",
}

EXCLUDE_FILES = {
    ".env",
    ".DS_Store",
    "submission.zip",
}

EXCLUDE_SUFFIXES = {".pyc"}


def _should_exclude(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    parts = set(rel.parts)

    if parts & EXCLUDE_DIRS:
        return True

    if path.name in EXCLUDE_FILES:
        return True

    if path.suffix in EXCLUDE_SUFFIXES:
        return True

    return False


def make_zip() -> Path:
    root = Path(__file__).resolve().parents[1]
    out_zip = root / "submission.zip"

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in root.rglob("*"):
            if path.is_dir():
                continue
            if _should_exclude(path, root):
                continue
            zf.write(path, path.relative_to(root).as_posix())

    return out_zip


def main() -> int:
    out_zip = make_zip()
    print(f"Created {out_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
