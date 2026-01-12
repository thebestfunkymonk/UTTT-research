"""
Random Agent

A simple baseline that selects moves uniformly at random.
"""
import random
from typing import Tuple, Optional, TYPE_CHECKING

from .base import Agent

if TYPE_CHECKING:
    from ..engine.state import UTTTState
    from ..engine.rules import UTTTRules


class RandomAgent(Agent):
    """
    Agent that selects moves uniformly at random from legal moves.
    
    Useful as a baseline and for generating random playouts in MCTS.
    """
    
    def __init__(self, seed: Optional[int] = None, name: Optional[str] = None):
        """
        Initialize random agent.
        
        Args:
            seed: Random seed for reproducibility
            name: Optional custom name
        """
        self._name = name or "Random"
        self._rng = random.Random(seed)
        self._seed = seed
    
    @property
    def name(self) -> str:
        return self._name
    
    def select_move(
        self, 
        state: 'UTTTState', 
        rules: 'UTTTRules'
    ) -> Tuple[int, int]:
        """Select a random legal move."""
        legal_moves = rules.get_legal_moves(state)
        if not legal_moves:
            raise ValueError("No legal moves available")
        return self._rng.choice(legal_moves)
    
    def reset(self):
        """Reset RNG to seed for reproducibility across games."""
        if self._seed is not None:
            self._rng = random.Random(self._seed)
