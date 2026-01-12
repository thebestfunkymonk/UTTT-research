"""
Self-Play Data Generation

Generates training data by having agents play against themselves.
This is the core of AlphaZero-style training.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, TYPE_CHECKING
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

if TYPE_CHECKING:
    from ..engine.state import UTTTState, Player
    from ..engine.rules import UTTTRules
    from ..agents.base import Agent


@dataclass
class GameRecord:
    """Record of a complete game for training."""
    states: List['UTTTState'] = field(default_factory=list)
    policies: List[Dict[Tuple[int, int], float]] = field(default_factory=list)
    result: Optional[float] = None  # 1 = X wins, 0 = O wins, 0.5 = draw
    
    def get_training_examples(self) -> List[Tuple['UTTTState', Dict, float]]:
        """
        Convert game record to training examples.
        
        Returns list of (state, policy, value) tuples where value is from
        the perspective of the player to move.
        """
        if self.result is None:
            return []
        
        examples = []
        from ..engine.state import Player
        
        for state, policy in zip(self.states, self.policies):
            # Value from perspective of current player
            if state.current_player == Player.X:
                value = self.result
            else:
                value = 1 - self.result
            
            examples.append((state, policy, value))
        
        return examples


class SelfPlayWorker:
    """
    Worker that generates games through self-play.
    
    Uses an agent (typically MCTS with NN) to play against itself,
    recording states, policies, and outcomes.
    """
    
    def __init__(
        self,
        agent: 'Agent',
        rules: 'UTTTRules',
        temperature_schedule: Optional[Dict[int, float]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize self-play worker.
        
        Args:
            agent: Agent to use for self-play
            rules: Game rules
            temperature_schedule: Dict mapping move number to temperature
                                 Default: 1.0 for first 30 moves, 0 after
            seed: Random seed
        """
        self.agent = agent
        self.rules = rules
        self.temperature_schedule = temperature_schedule or {0: 1.0, 30: 0.0}
        self._rng = random.Random(seed)
    
    def get_temperature(self, move_num: int) -> float:
        """Get temperature for a given move number."""
        temp = 1.0
        for threshold, t in sorted(self.temperature_schedule.items()):
            if move_num >= threshold:
                temp = t
        return temp
    
    def play_game(self) -> GameRecord:
        """
        Play one game of self-play.
        
        Returns:
            GameRecord with states, policies, and result
        """
        record = GameRecord()
        state = self.rules.create_initial_state()
        self.agent.reset()
        
        while not state.is_terminal():
            # Get policy from agent
            policy = self.agent.get_policy(state, self.rules)
            
            if policy is None:
                # Agent doesn't provide policy, use uniform
                legal_moves = self.rules.get_legal_moves(state)
                policy = {m: 1.0/len(legal_moves) for m in legal_moves}
            
            # Record state and policy
            record.states.append(state.copy())
            record.policies.append(policy)
            
            # Select move based on temperature
            temp = self.get_temperature(len(record.states))
            move = self._select_move_with_temperature(policy, temp)
            
            # Apply move
            state = self.rules.apply_move(state, move)
        
        # Record result
        from ..engine.state import Player
        winner = state.get_winner()
        if winner == Player.X:
            record.result = 1.0
        elif winner == Player.O:
            record.result = 0.0
        else:
            record.result = 0.5
        
        return record
    
    def _select_move_with_temperature(
        self, 
        policy: Dict[Tuple[int, int], float],
        temperature: float
    ) -> Tuple[int, int]:
        """Select move from policy with temperature."""
        moves = list(policy.keys())
        probs = np.array([policy[m] for m in moves])
        
        if temperature == 0:
            # Greedy
            return moves[np.argmax(probs)]
        
        # Apply temperature
        probs = np.power(probs, 1.0 / temperature)
        probs = probs / np.sum(probs)
        
        idx = self._rng.choices(range(len(moves)), weights=probs)[0]
        return moves[idx]


def generate_self_play_games(
    agent: 'Agent',
    rules: 'UTTTRules',
    num_games: int,
    num_workers: int = 1,
    temperature_schedule: Optional[Dict[int, float]] = None,
    seed: Optional[int] = None,
    progress_callback=None
) -> List[GameRecord]:
    """
    Generate multiple self-play games.
    
    Args:
        agent: Agent to use (should be thread-safe if num_workers > 1)
        rules: Game rules
        num_games: Number of games to generate
        num_workers: Number of parallel workers
        temperature_schedule: Temperature schedule for move selection
        seed: Base random seed
        progress_callback: Optional callback(games_completed, total)
    
    Returns:
        List of GameRecord objects
    """
    records = []
    
    if num_workers == 1:
        # Single-threaded
        worker = SelfPlayWorker(
            agent, rules, temperature_schedule, seed
        )
        for i in range(num_games):
            record = worker.play_game()
            records.append(record)
            if progress_callback:
                progress_callback(i + 1, num_games)
    else:
        # Multi-threaded (note: agent must be thread-safe)
        # For NN agents, typically create separate agents per worker
        def play_one_game(game_idx):
            worker_seed = seed + game_idx if seed else None
            worker = SelfPlayWorker(
                agent, rules, temperature_schedule, worker_seed
            )
            return worker.play_game()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(play_one_game, i) for i in range(num_games)]
            completed = 0
            for future in as_completed(futures):
                records.append(future.result())
                completed += 1
                if progress_callback:
                    progress_callback(completed, num_games)
    
    return records


@dataclass
class ReplayBuffer:
    """
    Buffer to store training examples from self-play games.
    
    Implements a FIFO buffer that stores the most recent examples.
    """
    max_size: int = 100000
    examples: List[Tuple['UTTTState', Dict, float]] = field(default_factory=list)
    
    def add_game(self, record: GameRecord):
        """Add examples from a game record."""
        examples = record.get_training_examples()
        self.examples.extend(examples)
        
        # Trim if over max size
        if len(self.examples) > self.max_size:
            self.examples = self.examples[-self.max_size:]
    
    def add_games(self, records: List[GameRecord]):
        """Add examples from multiple games."""
        for record in records:
            self.add_game(record)
    
    def sample(self, batch_size: int) -> List[Tuple['UTTTState', Dict, float]]:
        """Sample a batch of training examples."""
        if len(self.examples) < batch_size:
            return self.examples.copy()
        return random.sample(self.examples, batch_size)
    
    def __len__(self):
        return len(self.examples)
