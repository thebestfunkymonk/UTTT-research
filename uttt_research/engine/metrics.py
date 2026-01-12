"""
Metrics and Analytics for UTTT Research

This module provides helpers to calculate various game metrics for evaluating
rule variants:
- Fairness (win rates)
- Decisiveness (draw rates)  
- Move Freedom (average legal moves per turn)
- Board Utilization (distribution of moves across macro-boards)
- Drama (lead swings in win probability)
- Constraint Intensity (forced move sequences)
- Game Length
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
import numpy as np
from collections import Counter

if TYPE_CHECKING:
    from .state import UTTTState, Player
    from .rules import UTTTRules


@dataclass
class GameMetrics:
    """Metrics collected from a single game."""
    winner: Optional['Player'] = None
    game_length: int = 0
    move_freedoms: List[int] = field(default_factory=list)  # Legal moves at each turn
    macro_board_moves: List[int] = field(default_factory=list)  # Which macro-board each move was in
    value_history: List[float] = field(default_factory=list)  # Win probability estimates
    constraint_runs: List[int] = field(default_factory=list)  # Lengths of forced-target sequences
    
    @property
    def avg_move_freedom(self) -> float:
        """Average number of legal moves per turn."""
        return np.mean(self.move_freedoms) if self.move_freedoms else 0.0
    
    @property
    def board_utilization_entropy(self) -> float:
        """
        Entropy of move distribution across macro-boards.
        Higher = more evenly distributed.
        """
        if not self.macro_board_moves:
            return 0.0
        
        counts = Counter(self.macro_board_moves)
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        
        # Shannon entropy (base 2)
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        
        # Normalize by max possible entropy (log2(9) for 9 boards)
        max_entropy = np.log2(9)
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    @property
    def drama_score(self) -> float:
        """
        Measure of how much the game "swung" between players.
        Calculated as the sum of absolute value changes.
        """
        if len(self.value_history) < 2:
            return 0.0
        
        changes = np.abs(np.diff(self.value_history))
        return float(np.sum(changes))
    
    @property
    def lead_changes(self) -> int:
        """Number of times the leading player changed."""
        if len(self.value_history) < 2:
            return 0
        
        # Values > 0.5 favor current player, < 0.5 favor opponent
        leaders = [1 if v > 0.5 else (-1 if v < 0.5 else 0) for v in self.value_history]
        changes = 0
        last_leader = 0
        for leader in leaders:
            if leader != 0 and leader != last_leader:
                if last_leader != 0:
                    changes += 1
                last_leader = leader
        return changes
    
    @property
    def avg_constraint_intensity(self) -> float:
        """Average length of forced-target sequences."""
        return np.mean(self.constraint_runs) if self.constraint_runs else 1.0


@dataclass  
class VariantStatistics:
    """Aggregated statistics for a rule variant across many games."""
    variant_name: str
    games: List[GameMetrics] = field(default_factory=list)
    
    @property
    def num_games(self) -> int:
        return len(self.games)
    
    def win_rate(self, player: 'Player') -> float:
        """Win rate for a specific player."""
        if not self.games:
            return 0.0
        wins = sum(1 for g in self.games if g.winner == player)
        return wins / len(self.games)
    
    @property
    def draw_rate(self) -> float:
        """Proportion of games ending in draw."""
        if not self.games:
            return 0.0
        draws = sum(1 for g in self.games if g.winner is None)
        return draws / len(self.games)
    
    @property
    def fairness_score(self) -> float:
        """
        Measure of balance (0 = perfectly fair, 1 = completely unfair).
        Based on deviation from 50/50 win rate (excluding draws).
        """
        from .state import Player
        
        decisive_games = [g for g in self.games if g.winner is not None]
        if not decisive_games:
            return 0.0
        
        x_wins = sum(1 for g in decisive_games if g.winner == Player.X)
        x_rate = x_wins / len(decisive_games)
        
        # Distance from 0.5
        return abs(x_rate - 0.5) * 2
    
    @property
    def avg_game_length(self) -> float:
        """Average number of moves per game."""
        if not self.games:
            return 0.0
        return np.mean([g.game_length for g in self.games])
    
    @property
    def std_game_length(self) -> float:
        """Standard deviation of game length."""
        if not self.games:
            return 0.0
        return np.std([g.game_length for g in self.games])
    
    @property
    def avg_move_freedom(self) -> float:
        """Average move freedom across all games."""
        if not self.games:
            return 0.0
        return np.mean([g.avg_move_freedom for g in self.games])
    
    @property
    def avg_board_utilization(self) -> float:
        """Average board utilization entropy."""
        if not self.games:
            return 0.0
        return np.mean([g.board_utilization_entropy for g in self.games])
    
    @property
    def avg_drama(self) -> float:
        """Average drama score."""
        if not self.games:
            return 0.0
        return np.mean([g.drama_score for g in self.games])
    
    @property
    def avg_lead_changes(self) -> float:
        """Average number of lead changes."""
        if not self.games:
            return 0.0
        return np.mean([g.lead_changes for g in self.games])
    
    @property
    def avg_constraint_intensity(self) -> float:
        """Average constraint intensity."""
        if not self.games:
            return 0.0
        return np.mean([g.avg_constraint_intensity for g in self.games])
    
    def summary(self) -> Dict:
        """Get a summary dictionary of all metrics."""
        from .state import Player
        
        return {
            'variant': self.variant_name,
            'games': self.num_games,
            'x_win_rate': self.win_rate(Player.X),
            'o_win_rate': self.win_rate(Player.O),
            'draw_rate': self.draw_rate,
            'fairness': 1 - self.fairness_score,  # Invert so higher = better
            'avg_game_length': self.avg_game_length,
            'std_game_length': self.std_game_length,
            'avg_move_freedom': self.avg_move_freedom,
            'avg_board_utilization': self.avg_board_utilization,
            'avg_drama': self.avg_drama,
            'avg_lead_changes': self.avg_lead_changes,
            'avg_constraint_intensity': self.avg_constraint_intensity,
        }
    
    def __str__(self) -> str:
        """Human-readable summary."""
        s = self.summary()
        lines = [
            f"=== {s['variant']} ({s['games']} games) ===",
            f"  Win Rates: X={s['x_win_rate']*100:.1f}%, O={s['o_win_rate']*100:.1f}%, Draw={s['draw_rate']*100:.1f}%",
            f"  Fairness: {s['fairness']*100:.1f}%",
            f"  Game Length: {s['avg_game_length']:.1f} ± {s['std_game_length']:.1f} moves",
            f"  Move Freedom: {s['avg_move_freedom']:.1f} choices/turn",
            f"  Board Utilization: {s['avg_board_utilization']*100:.1f}%",
            f"  Drama: {s['avg_drama']:.2f} (avg swings)",
            f"  Lead Changes: {s['avg_lead_changes']:.1f} per game",
            f"  Constraint Intensity: {s['avg_constraint_intensity']:.1f} moves/sequence",
        ]
        return "\n".join(lines)


class MetricsCollector:
    """Helper class to collect metrics during game play."""
    
    def __init__(self, value_estimator=None):
        """
        Initialize collector.
        
        Args:
            value_estimator: Optional callable(state) -> float that estimates
                           win probability for current player. If None, value
                           tracking is disabled.
        """
        self.value_estimator = value_estimator
        self.reset()
    
    def reset(self):
        """Reset for a new game."""
        self._move_freedoms = []
        self._macro_boards = []
        self._values = []
        self._constraint_runs = []
        self._current_constraint_run = 0
        self._last_target = None
    
    def record_turn(self, state: 'UTTTState', legal_moves: List[Tuple[int, int]]):
        """Record metrics for a turn before move is made."""
        # Move freedom
        self._move_freedoms.append(len(legal_moves))
        
        # Value estimation
        if self.value_estimator:
            value = self.value_estimator(state)
            self._values.append(value)
        
        # Constraint tracking
        if state.target_macro is not None:
            if state.target_macro == self._last_target:
                self._current_constraint_run += 1
            else:
                if self._current_constraint_run > 0:
                    self._constraint_runs.append(self._current_constraint_run)
                self._current_constraint_run = 1
            self._last_target = state.target_macro
        else:
            if self._current_constraint_run > 0:
                self._constraint_runs.append(self._current_constraint_run)
            self._current_constraint_run = 0
            self._last_target = None
    
    def record_move(self, move: Tuple[int, int]):
        """Record the move that was made."""
        macro_idx, cell_idx = move
        if macro_idx >= 0:  # Not a special move like swap
            self._macro_boards.append(macro_idx)
    
    def finalize(self, state: 'UTTTState') -> GameMetrics:
        """Finalize and return metrics for completed game."""
        # Record final constraint run
        if self._current_constraint_run > 0:
            self._constraint_runs.append(self._current_constraint_run)
        
        return GameMetrics(
            winner=state.get_winner(),
            game_length=len(state.move_history),
            move_freedoms=self._move_freedoms.copy(),
            macro_board_moves=self._macro_boards.copy(),
            value_history=self._values.copy(),
            constraint_runs=self._constraint_runs.copy() if self._constraint_runs else [1],
        )


def compare_variants(stats_list: List[VariantStatistics]) -> str:
    """Generate a comparison report for multiple variants."""
    lines = ["=" * 60, "VARIANT COMPARISON REPORT", "=" * 60, ""]
    
    # Header
    headers = ["Metric"] + [s.variant_name[:15] for s in stats_list]
    col_width = 16
    lines.append("".join(h.ljust(col_width) for h in headers))
    lines.append("-" * (col_width * len(headers)))
    
    # Metrics to compare
    metrics = [
        ("X Win %", lambda s: f"{s.win_rate(s.games[0].winner.__class__.X)*100:.1f}%" if s.games else "N/A"),
        ("O Win %", lambda s: f"{s.win_rate(s.games[0].winner.__class__.O)*100:.1f}%" if s.games else "N/A"),
        ("Draw %", lambda s: f"{s.draw_rate*100:.1f}%"),
        ("Fairness", lambda s: f"{(1-s.fairness_score)*100:.1f}%"),
        ("Avg Length", lambda s: f"{s.avg_game_length:.1f}"),
        ("Move Freedom", lambda s: f"{s.avg_move_freedom:.1f}"),
        ("Utilization", lambda s: f"{s.avg_board_utilization*100:.1f}%"),
        ("Drama", lambda s: f"{s.avg_drama:.2f}"),
        ("Lead Changes", lambda s: f"{s.avg_lead_changes:.1f}"),
        ("Constraint", lambda s: f"{s.avg_constraint_intensity:.1f}"),
    ]
    
    from .state import Player
    
    for name, func in metrics:
        row = [name]
        for stats in stats_list:
            try:
                if "Win %" in name:
                    p = Player.X if "X" in name else Player.O
                    row.append(f"{stats.win_rate(p)*100:.1f}%")
                else:
                    row.append(func(stats))
            except:
                row.append("N/A")
        lines.append("".join(str(v).ljust(col_width) for v in row))
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)
