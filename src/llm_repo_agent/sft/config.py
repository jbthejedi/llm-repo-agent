from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SFTExtractConfig:
    """Configuration for SFT data extraction."""
    trace_dir: Path
    output_path: Path
    require_success: bool = True
    require_valid_tool_ok: bool = True
    max_context_chars: int = 8000
    include_recovery: bool = False
    output_format: str = "json"  # "json" | "native"
    progress: bool = True
