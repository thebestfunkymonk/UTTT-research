"""
Evaluation and Arena for Agent Comparison

Provides tools to evaluate agents against each other and track performance.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
import time

if TYPE_CHECKING:
    from ..engine.state import UTTTState, Player
    from ..engine.rules import UTTTRules
    from ..agents.base import Agent
    from ..engine.metrics import GameMetrics, MetricsCollector


@dataclass
class MatchResult:
    """Result of a single match between two agents."""
    player_x: str
    player_o: str
    winner: Optional[str]  # Name of winner, or None for draw
    num_moves: int
    duration_seconds: float
    x_value_estimates: List[float] = field(default_factory=list)
    o_value_estimates: List[float] = field(default_factory=list)


@dataclass
class HeadToHeadStats:
    """Statistics for matches between two specific agents."""
    agent_a: str
    agent_b: str
    a_wins_as_x: int = 0
    a_wins_as_o: int = 0
    b_wins_as_x: int = 0
    b_wins_as_o: int = 0
    draws: int = 0
    total_games: int = 0
    
    @property
    def a_wins(self) -> int:
        return self.a_wins_as_x + self.a_wins_as_o
    
    @property
    def b_wins(self) -> int:
        return self.b_wins_as_x + self.b_wins_as_o
    
    @property
    def a_win_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return self.a_wins / self.total_games
    
    def __str__(self) -> str:
        return (
            f"{self.agent_a} vs {self.agent_b}: "
            f"{self.a_wins}-{self.b_wins}-{self.draws} "
            f"({self.a_win_rate*100:.1f}% win rate for {self.agent_a})"
        )


class Arena:
    """
    Arena for running matches between agents.
    """
    
    def __init__(self, rules: 'UTTTRules', collect_metrics: bool = True):
        """
        Initialize arena.
        
        Args:
            rules: Game rules to use
            collect_metrics: Whether to collect detailed metrics
        """
        self.rules = rules
        self.collect_metrics = collect_metrics
        self.match_history: List[MatchResult] = []
    
    def play_match(
        self, 
        agent_x: 'Agent', 
        agent_o: 'Agent',
        verbose: bool = False
    ) -> MatchResult:
        """
        Play a single match between two agents.
        
        Args:
            agent_x: Agent playing as X (first player)
            agent_o: Agent playing as O (second player)
            verbose: Whether to print moves
            
        Returns:
            MatchResult with outcome and statistics
        """
        from ..engine.state import Player
        
        state = self.rules.create_initial_state()
        agent_x.reset()
        agent_o.reset()
        
        x_values = []
        o_values = []
        
        start_time = time.time()
        
        while not state.is_terminal():
            current_agent = agent_x if state.current_player == Player.X else agent_o
            
            # Get move
            move = current_agent.select_move(state, self.rules)
            
            # Record value estimates
            if state.current_player == Player.X:
                val = agent_x.get_value_estimate(state)
                if val is not None:
                    x_values.append(val)
            else:
                val = agent_o.get_value_estimate(state)
                if val is not None:
                    o_values.append(val)
            
            if verbose:
                print(f"Move {len(state.move_history)+1}: {state.current_player} plays {move}")
            
            # Apply move and notify opponent
            state = self.rules.apply_move(state, move)
            
            if state.current_player == Player.X:
                agent_x.on_opponent_move(move, state)
            else:
                agent_o.on_opponent_move(move, state)
        
        duration = time.time() - start_time
        
        # Determine winner
        winner_player = state.get_winner()
        if winner_player == Player.X:
            winner = agent_x.name
        elif winner_player == Player.O:
            winner = agent_o.name
        else:
            winner = None
        
        result = MatchResult(
            player_x=agent_x.name,
            player_o=agent_o.name,
            winner=winner,
            num_moves=len(state.move_history),
            duration_seconds=duration,
            x_value_estimates=x_values,
            o_value_estimates=o_values,
        )
        
        self.match_history.append(result)
        
        if verbose:
            print(f"\nGame over! Winner: {winner or 'Draw'} ({result.num_moves} moves, {duration:.2f}s)")
        
        return result
    
    def play_matches(
        self,
        agent_a: 'Agent',
        agent_b: 'Agent',
        num_games: int,
        alternate: bool = True,
        verbose: bool = False,
        progress_callback=None
    ) -> HeadToHeadStats:
        """
        Play multiple matches between two agents.
        
        Args:
            agent_a: First agent
            agent_b: Second agent
            num_games: Number of games to play
            alternate: Whether to alternate who plays X
            verbose: Whether to print each game
            progress_callback: Optional callback(games_played, total)
            
        Returns:
            HeadToHeadStats with aggregate results
        """
        stats = HeadToHeadStats(agent_a=agent_a.name, agent_b=agent_b.name)
        
        for i in range(num_games):
            # Determine who plays X
            if alternate and i % 2 == 1:
                agent_x, agent_o = agent_b, agent_a
                a_is_x = False
            else:
                agent_x, agent_o = agent_a, agent_b
                a_is_x = True
            
            result = self.play_match(agent_x, agent_o, verbose=verbose)
            stats.total_games += 1
            
            # Update stats
            if result.winner is None:
                stats.draws += 1
            elif result.winner == agent_a.name:
                if a_is_x:
                    stats.a_wins_as_x += 1
                else:
                    stats.a_wins_as_o += 1
            else:
                if a_is_x:
                    stats.b_wins_as_o += 1
                else:
                    stats.b_wins_as_x += 1
            
            if progress_callback:
                progress_callback(i + 1, num_games)
        
        return stats


class Evaluator:
    """
    Evaluator for comparing multiple agents and rule variants.
    """
    
    def __init__(self):
        self.results: Dict[str, Dict[str, HeadToHeadStats]] = {}
    
    def evaluate_agents(
        self,
        agents: List['Agent'],
        rules: 'UTTTRules',
        games_per_pair: int = 100,
        verbose: bool = True
    ) -> Dict[str, Dict[str, HeadToHeadStats]]:
        """
        Round-robin evaluation of multiple agents.
        
        Args:
            agents: List of agents to evaluate
            rules: Game rules
            games_per_pair: Games per agent pair
            verbose: Whether to print progress
            
        Returns:
            Dict mapping (agent_a, agent_b) to HeadToHeadStats
        """
        arena = Arena(rules)
        results = {}
        
        for i, agent_a in enumerate(agents):
            results[agent_a.name] = {}
            for j, agent_b in enumerate(agents):
                if i >= j:  # Skip self-play and already computed pairs
                    continue
                
                if verbose:
                    print(f"\n{agent_a.name} vs {agent_b.name}...")
                
                def progress(done, total):
                    if verbose and done % 10 == 0:
                        print(f"  {done}/{total} games")
                
                stats = arena.play_matches(
                    agent_a, agent_b, games_per_pair,
                    progress_callback=progress
                )
                
                results[agent_a.name][agent_b.name] = stats
                
                if verbose:
                    print(f"  Result: {stats}")
        
        self.results = results
        return results
    
    def evaluate_variant(
        self,
        agent: 'Agent',
        rules: 'UTTTRules',
        num_games: int = 100,
        verbose: bool = True
    ):
        """
        Evaluate a rule variant using self-play.
        
        Args:
            agent: Agent to use for evaluation
            rules: Rule variant to evaluate
            num_games: Number of self-play games
            verbose: Whether to print progress
            
        Returns:
            VariantStatistics object
        """
        from ..engine.metrics import VariantStatistics, MetricsCollector, GameMetrics
        from ..engine.state import Player
        
        stats = VariantStatistics(variant_name=rules.name)
        arena = Arena(rules)
        
        # Create value estimator from agent
        def value_estimator(state):
            val = agent.get_value_estimate(state)
            return val if val is not None else 0.5
        
        for i in range(num_games):
            # Self-play game
            agent.reset()
            collector = MetricsCollector(value_estimator)
            state = self.rules.create_initial_state()
            
            while not state.is_terminal():
                legal_moves = rules.get_legal_moves(state)
                collector.record_turn(state, legal_moves)
                
                move = agent.select_move(state, rules)
                collector.record_move(move)
                
                state = rules.apply_move(state, move)
            
            metrics = collector.finalize(state)
            stats.games.append(metrics)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  {i+1}/{num_games} games")
        
        return stats
    
    def summary_table(self) -> str:
        """Generate a summary table of all results."""
        lines = ["=" * 60, "EVALUATION SUMMARY", "=" * 60]
        
        for agent_a, opponents in self.results.items():
            for agent_b, stats in opponents.items():
                lines.append(str(stats))
        
        lines.append("=" * 60)
        return "\n".join(lines)


def quick_evaluate(
    agent_a: 'Agent',
    agent_b: 'Agent',
    rules: 'UTTTRules',
    num_games: int = 100,
    verbose: bool = True
) -> HeadToHeadStats:
    """
    Quick evaluation of two agents.
    
    Args:
        agent_a: First agent
        agent_b: Second agent
        rules: Game rules
        num_games: Number of games
        verbose: Whether to print progress
        
    Returns:
        HeadToHeadStats
    """
    arena = Arena(rules)
    
    def progress(done, total):
        if verbose and done % (total // 10 or 1) == 0:
            print(f"Progress: {done}/{total}")
    
    stats = arena.play_matches(
        agent_a, agent_b, num_games,
        progress_callback=progress if verbose else None
    )
    
    if verbose:
        print(f"\nFinal: {stats}")
    
    return stats
