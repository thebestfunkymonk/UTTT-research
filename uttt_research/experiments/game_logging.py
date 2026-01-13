"""
Study game logging utilities.

Provides JSONL outputs for raw analysis and an LLM-friendly variant.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from ..engine.state import MacroBoardStatus, Player


RAW_FORMAT_VERSION = "1.0"
LLM_FORMAT_VERSION = "1.0"


def _player_symbol(value: int) -> str:
    if value == Player.X:
        return "X"
    if value == Player.O:
        return "O"
    return "."


def format_macro_board_rows(macro_board: np.ndarray) -> List[str]:
    """Format macro-board status as row strings."""
    rows = []
    for row in range(3):
        row_chars = []
        for col in range(3):
            status = MacroBoardStatus(macro_board[row, col])
            if status == MacroBoardStatus.X_WON:
                row_chars.append("X")
            elif status == MacroBoardStatus.O_WON:
                row_chars.append("O")
            elif status == MacroBoardStatus.DRAW:
                row_chars.append("D")
            else:
                row_chars.append(".")
        rows.append("".join(row_chars))
    return rows


def format_board_rows(board: np.ndarray) -> List[str]:
    """Format full 9x9 board as row strings."""
    rows = []
    for row in range(9):
        rows.append("".join(_player_symbol(int(value)) for value in board[row]))
    return rows


@dataclass
class StudyGameLogger:
    """Write game records to JSONL files with a manifest."""
    output_dir: Path
    study_id: str
    config: Dict[str, Any]
    write_llm: bool = True
    raw_path: Path = field(init=False)
    llm_path: Optional[Path] = field(init=False, default=None)
    manifest_path: Path = field(init=False)
    _raw_file: Optional[Any] = field(init=False, default=None)
    _llm_file: Optional[Any] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_path = self.output_dir / f"{self.study_id}_games.jsonl"
        self.llm_path = (
            self.output_dir / f"{self.study_id}_games_llm.jsonl"
            if self.write_llm
            else None
        )
        self.manifest_path = self.output_dir / f"{self.study_id}_games_manifest.json"

        manifest = {
            "study_id": self.study_id,
            "created_at": self.config.get("timestamp"),
            "raw_format_version": RAW_FORMAT_VERSION,
            "llm_format_version": LLM_FORMAT_VERSION if self.write_llm else None,
            "files": {
                "raw": str(self.raw_path),
                "llm": str(self.llm_path) if self.llm_path else None,
            },
            "config": self.config,
        }
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        self._raw_file = self.raw_path.open("w", encoding="utf-8")
        if self.llm_path:
            self._llm_file = self.llm_path.open("w", encoding="utf-8")

    def record_game(self, raw_record: Dict[str, Any], llm_record: Optional[Dict[str, Any]]) -> None:
        """Write a single game record to disk."""
        if not self._raw_file:
            raise RuntimeError("Logger not initialized.")
        self._raw_file.write(json.dumps(raw_record) + "\n")
        if self._llm_file and llm_record is not None:
            self._llm_file.write(json.dumps(llm_record) + "\n")

    def close(self) -> None:
        """Close file handles."""
        if self._raw_file:
            self._raw_file.close()
        if self._llm_file:
            self._llm_file.close()

    def __enter__(self) -> "StudyGameLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def build_move_sequence(moves: Iterable[Dict[str, Any]]) -> str:
    """Build a compact move sequence string for LLM parsing."""
    parts = []
    for move in moves:
        target = move.get("target_macro")
        target_str = "*" if target is None else str(target)
        parts.append(f"{move['turn']}:{move['player']} {move['macro_idx']}/{move['cell_idx']}->{target_str}")
    return " | ".join(parts)
