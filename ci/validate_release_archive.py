#!/usr/bin/env python3
"""Validate a release tarball before extracting it into a build or runtime tree.

The archives used by this project are produced by our packaging code, but they cross a
release boundary.  Do not let a corrupt or replaced archive turn a path traversal, link,
or an incomplete serving artifact into a successful deployment.  This helper deliberately
only validates; callers can extract with their normal tar tool after it exits zero.
"""

from __future__ import annotations

import argparse
import posixpath
import sys
import tarfile
from pathlib import Path
from typing import Iterable


class ArchiveError(ValueError):
    """Raised when an archive is unsafe or incomplete."""


def _normalise_member_name(name: str) -> str:
    """Return a safe, canonical POSIX archive member name or reject it."""
    if not name or "\x00" in name:
        raise ArchiveError("archive contains an empty or NUL-containing path")
    if "\\" in name:
        raise ArchiveError(f"archive member uses a backslash path: {name!r}")
    if name.startswith("/") or posixpath.isabs(name):
        raise ArchiveError(f"archive member is absolute: {name!r}")
    if len(name) >= 2 and name[1] == ":" and name[0].isalpha():
        raise ArchiveError(f"archive member uses a drive path: {name!r}")

    parts = name.split("/")
    if ".." in parts:
        raise ArchiveError(f"archive member escapes its extraction root: {name!r}")

    normalised = posixpath.normpath(name)
    if normalised in ("", ".") or normalised == ".." or normalised.startswith("../"):
        raise ArchiveError(f"archive member has an unsafe path: {name!r}")
    if normalised.startswith("/"):
        raise ArchiveError(f"archive member is absolute after normalisation: {name!r}")
    return normalised


def _normalise_prefix(value: str) -> str:
    prefix = _normalise_member_name(value.rstrip("/"))
    return prefix + "/"


def _normalise_required_file(value: str) -> str:
    return _normalise_member_name(value)


def _is_allowed(
    name: str,
    allow_prefixes: Iterable[str],
    allow_files: Iterable[str],
) -> bool:
    if not allow_prefixes and not allow_files:
        return True
    if name in allow_files:
        return True
    return any(name == prefix[:-1] or name.startswith(prefix) for prefix in allow_prefixes)


def validate_archive(
    archive_path: Path,
    *,
    allow_prefixes: Iterable[str] = (),
    allow_files: Iterable[str] = (),
    required_files: Iterable[str] = (),
    required_prefixes: Iterable[str] = (),
) -> int:
    """Validate *archive_path* and return the number of members inspected.

    ``tarfile`` is used only for reading metadata.  Regular files and directories are
    allowed; links and special files are rejected because they can redirect a later
    extraction or overwrite data outside the intended tree.
    """
    allow_prefixes = tuple(allow_prefixes)
    allow_files = tuple(allow_files)
    required_files = tuple(required_files)
    required_prefixes = tuple(required_prefixes)

    # Validate command-line path constraints before opening the archive as well, so a
    # malformed CI invocation fails just as clearly as a malformed archive.
    try:
        allow_prefixes = tuple(_normalise_prefix(value) for value in allow_prefixes)
        allow_files = tuple(_normalise_required_file(value) for value in allow_files)
        required_files = tuple(_normalise_required_file(value) for value in required_files)
        required_prefixes = tuple(_normalise_prefix(value) for value in required_prefixes)
        if archive_path.is_symlink():
            raise ArchiveError("archive path is a symlink")
    except (ArchiveError, OSError) as exc:
        print(f"release archive rejected: {exc}", file=sys.stderr)
        return 1

    names: set[str] = set()
    file_names: set[str] = set()
    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            for member in archive:
                name = _normalise_member_name(member.name)
                if name in names:
                    raise ArchiveError(f"archive contains duplicate member: {name!r}")
                if member.issym() or member.islnk():
                    raise ArchiveError(f"archive contains a link member: {name!r}")
                if not (member.isfile() or member.isdir()):
                    raise ArchiveError(f"archive contains a special member: {name!r}")
                if not _is_allowed(name, allow_prefixes, allow_files):
                    raise ArchiveError(f"archive member is outside the allowed paths: {name!r}")
                names.add(name)
                if member.isfile():
                    file_names.add(name)
    except (ArchiveError, OSError, EOFError, tarfile.TarError) as exc:
        print(f"release archive rejected: {exc}", file=sys.stderr)
        return 1

    if not names:
        print("release archive rejected: it contains no members", file=sys.stderr)
        return 1

    missing = [value for value in required_files if value not in file_names]
    if missing:
        print(
            "release archive rejected: missing required file member(s): "
            + ", ".join(repr(value) for value in missing),
            file=sys.stderr,
        )
        return 1

    for prefix in required_prefixes:
        if not any(name.startswith(prefix) for name in file_names):
            print(
                f"release archive rejected: no regular file under required prefix {prefix!r}",
                file=sys.stderr,
            )
            return 1

    print(f"release archive validated: {len(names)} member(s) from {archive_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="tar archive to inspect")
    parser.add_argument(
        "--allow-prefix",
        action="append",
        default=[],
        help="only allow members at or below this path (repeatable)",
    )
    parser.add_argument(
        "--allow-file",
        action="append",
        default=[],
        help="allow this exact file path (repeatable)",
    )
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        help="require this exact regular-file member (repeatable)",
    )
    parser.add_argument(
        "--require-prefix",
        action="append",
        default=[],
        help="require at least one regular file below this path (repeatable)",
    )
    args = parser.parse_args()

    return validate_archive(
        args.archive,
        allow_prefixes=args.allow_prefix,
        allow_files=args.allow_file,
        required_files=args.require,
        required_prefixes=args.require_prefix,
    )


if __name__ == "__main__":
    raise SystemExit(main())
