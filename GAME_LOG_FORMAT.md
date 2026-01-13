# Study Game Log Format

The variant study runner (`uttt_research.experiments.run_study`) can emit **per-game logs**
when invoked with `--output <dir> --save-games`.

Running with that flag produces three files (all keyed by the same `study_id` timestamp):

1. `study_<YYYYMMDD_HHMMSS>.json` — the study summary already produced today.
2. `<study_id>_games.jsonl` — **raw** per-game records (one JSON object per line).
3. `<study_id>_games_llm.jsonl` — **LLM-friendly** records (one JSON object per line).
4. `<study_id>_games_manifest.json` — metadata and format versions.

## Manifest (`*_games_manifest.json`)

```json
{
  "study_id": "20250101_123000",
  "created_at": "2025-01-01T12:30:00.000000",
  "raw_format_version": "1.0",
  "llm_format_version": "1.0",
  "files": {
    "raw": "outputs/20250101_123000_games.jsonl",
    "llm": "outputs/20250101_123000_games_llm.jsonl"
  },
  "config": {
    "timestamp": "...",
    "num_skill_games": 100,
    "mcts_simulations": 100,
    "randomness_levels": [0.0, 0.1, 0.25, 0.5],
    "seed": 42,
    "variants": ["Standard", "Balanced", "..."]
  }
}
```

## Raw Game Logs (`*_games.jsonl`)

Each line is a self-contained game record:

```json
{
  "game_id": "mcts_random_0.10:Standard:4",
  "variant": "Standard",
  "randomness_level": 0.1,
  "mcts_simulations": 100,
  "seed": 42,
  "agent": "MCTS(100, rand=0.10)",
  "moves": [
    {
      "turn": 1,
      "player": "X",
      "macro_idx": 4,
      "cell_idx": 0,
      "target_macro": null,
      "legal_moves": [[0, 0], [0, 1], "..."],
      "legal_moves_count": 81
    }
  ],
  "final_state": {
    "winner": "X",
    "macro_board": [[1, 0, 0], [0, 2, 0], [0, 0, 3]],
    "board": [[1, 0, "..."], "..."],
    "macro_board_rows": ["X..", ".O.", "..D"],
    "board_rows": ["X..O.....", "..."]
  },
  "metrics": {
    "winner": "X",
    "game_length": 61,
    "move_freedoms": [81, 9, "..."],
    "macro_board_moves": [4, 0, "..."],
    "value_history": [0.5, 0.51, "..."],
    "constraint_runs": [3, 2, "..."]
  }
}
```

**Notes**
- `macro_idx` / `cell_idx` follow the engine convention (0–8 in row-major order).
- `target_macro` is the constraint *before* the move is made.
- `board` is the full 9×9 grid of integers (0 = empty, 1 = X, 2 = O).
- `macro_board` is the 3×3 macro grid of status integers
  (`0 = ongoing, 1 = X won, 2 = O won, 3 = draw`).

## LLM Game Logs (`*_games_llm.jsonl`)

These records are simplified to emphasize sequence and outcomes:

```json
{
  "game_id": "mcts_random_0.10:Standard:4",
  "variant": "Standard",
  "randomness_level": 0.1,
  "winner": "X",
  "move_count": 61,
  "move_sequence": "1:X 4/0->* | 2:O 0/4->4 | ...",
  "moves": [
    { "turn": 1, "player": "X", "move": "4/0", "target_macro": null, "legal_moves_count": 81 }
  ],
  "macro_board_rows": ["X..", ".O.", "..D"],
  "board_rows": ["X..O.....", "..."]
}
```

The `move_sequence` string is built as:

```
<turn>:<player> <macro_idx>/<cell_idx>-><target_macro>
```

Where `target_macro` is `*` when the next player is unconstrained.
