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
    drop_postfix_on_loop: bool = False
    filter_write_file_targets: bool = False
    require_root_list_files_first: bool = False
    max_context_chars: int = 8000
    include_recovery: bool = False
    output_format: str = "json"  # "json" | "native"
    progress: bool = True
