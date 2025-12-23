from __future__ import annotations
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Sandbox:
  root: Path


def materialize_repo_sandbox(src: Path, dest: Optional[Path] = None) -> Sandbox:
  """Create a writable sandbox copy of the repo.

  If dest is provided, it must be empty or non-existent. Otherwise a temp dir is created.
  """
  src = src.expanduser().resolve()
  if dest is None:
    dest = Path(tempfile.mkdtemp(prefix="repo-agent-")).resolve()
  else:
    dest = dest.expanduser().resolve()
    if dest.exists() and any(dest.iterdir()):
      raise ValueError(f"sandbox destination is not empty: {dest}")
    dest.mkdir(parents=True, exist_ok=True)

  shutil.copytree(src, dest, dirs_exist_ok=True)
  return Sandbox(root=dest)


def cleanup_sandbox(sandbox: Sandbox) -> None:
  shutil.rmtree(sandbox.root, ignore_errors=True)
