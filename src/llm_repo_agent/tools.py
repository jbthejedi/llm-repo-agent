from __future__ import annotations
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional


@dataclass
class ToolResult:
  ok: bool
  output: str
  meta: Dict[str, Any]


class RepoTools:
  """
    Very intentional: we do NOT expose arbitrary shell execution.
    We expose a small allowlist of operations to reduce footguns.
    """

  def __init__(self, repo_root: Path):
    self.repo_root = repo_root.resolve()

  def _safe_path(self, rel: str) -> Path:
    p = (self.repo_root / rel).resolve()
    if self.repo_root not in p.parents and p != self.repo_root:
      raise ValueError("Path escapes repo_root.")
    return p

  def list_files(self, rel_dir: str = ".", max_files: int = 200) -> ToolResult:
    d = self._safe_path(rel_dir)
    if not d.exists():
      return ToolResult(False, f"Directory not found: {rel_dir}", {"rel_dir": rel_dir})
    files: List[str] = []
    for p in d.rglob("*"):
      if p.is_file():
        files.append(str(p.relative_to(self.repo_root)))
        if len(files) >= max_files:
          break
    return ToolResult(True, "\n".join(files), {"count": len(files), "rel_dir": rel_dir})

  def read_file(self, rel_path: str, max_chars: int = 12000) -> ToolResult:
    p = self._safe_path(rel_path)
    if not p.exists() or not p.is_file():
      return ToolResult(False, f"File not found: {rel_path}", {"rel_path": rel_path})
    txt = p.read_text(encoding="utf-8", errors="replace")
    truncated = txt[:max_chars]
    meta = {"rel_path": rel_path, "chars": len(txt), "truncated": len(txt) > max_chars}
    return ToolResult(True, truncated, meta)

  def write_file(self, rel_path: str, content: str) -> ToolResult:
    p = self._safe_path(rel_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return ToolResult(True, f"Wrote {rel_path} ({len(content)} chars).", {"rel_path": rel_path})

  def grep(self, pattern: str, rel_dir: str = ".", max_hits: int = 100) -> ToolResult:
    d = self._safe_path(rel_dir)
    if not d.exists():
      return ToolResult(False, f"Directory not found: {rel_dir}", {"rel_dir": rel_dir})

    hits: List[str] = []
    # simple python grep to stay portable
    for p in d.rglob("*"):
      if not p.is_file():
        continue
      try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
      except Exception:
        continue
      if pattern in txt:
        # record a few line snippets
        for i, line in enumerate(txt.splitlines(), start=1):
          if pattern in line:
            hits.append(f"{p.relative_to(self.repo_root)}:{i}:{line.strip()}")
            if len(hits) >= max_hits:
              return ToolResult(True, "\n".join(hits), {"pattern": pattern, "truncated": True})
    return ToolResult(True, "\n".join(hits) if hits else "(no matches)", {
        "pattern": pattern,
        "count": len(hits)
    })

  def run_tests(self, cmd: List[str], timeout_s: int = 120) -> ToolResult:
    """
        You choose the command (e.g. ["pytest","-q"] or ["python","-m","pytest","-q"]).
        Still not arbitrary: this runs a *single* command list you supply from your driver,
        not from the model.
        """
    try:
      proc = subprocess.run(
          cmd,
          cwd=str(self.repo_root),
          capture_output=True,
          text=True,
          timeout=timeout_s,
      )
      out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
      ok = proc.returncode == 0
      return ToolResult(ok, out.strip(), {"cmd": cmd, "returncode": proc.returncode})
    except subprocess.TimeoutExpired:
      return ToolResult(False, f"Timed out after {timeout_s}s running: {cmd}", {
          "cmd": cmd,
          "timeout_s": timeout_s
      })
