"""
Abstract Agent Interface

All agents must implement this interface to be compatible with the
research harness.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..engine.state import UTTTState
    from ..engine.rules import UTTTRules


class Agent(ABC):
    """
    Abstract base class for UTTT agents.
    
    Agents receive game states and rules, and return moves.
    They can optionally maintain internal state across moves.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this agent."""
        pass
    
    @abstractmethod
    def select_move(
        self, 
        state: 'UTTTState', 
        rules: 'UTTTRules'
    ) -> Tuple[int, int]:
        """
        Select a move given the current state.
        
        Args:
            state: Current game state
            rules: Rules being used (for getting legal moves, etc.)
            
        Returns:
            (macro_idx, cell_idx) tuple representing the chosen move
        """
        pass
    
    def reset(self):
        """
        Reset agent state for a new game.
        
        Override this if your agent maintains state across moves.
        """
        pass
    
    def get_value_estimate(self, state: 'UTTTState') -> Optional[float]:
        """
        Get the agent's estimated win probability for the current player.
        
        Override this if your agent can provide value estimates (e.g., MCTS, NN).
        
        Args:
            state: Current game state
            
        Returns:
            Float between 0 and 1, or None if not available
        """
        return None
    
    def get_policy(self, state: 'UTTTState', rules: 'UTTTRules') -> Optional[dict]:
        """
        Get the agent's move probability distribution.
        
        Override this if your agent can provide policy distributions.
        
        Args:
            state: Current game state
            rules: Rules being used
            
        Returns:
            Dict mapping (macro_idx, cell_idx) -> probability, or None
        """
        return None
    
    def on_opponent_move(self, move: Tuple[int, int], state: 'UTTTState'):
        """
        Called when the opponent makes a move.
        
        Override this to update internal state based on opponent's move.
        Useful for MCTS agents that can reuse tree search.
        
        Args:
            move: The move the opponent made
            state: The state after the move
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
