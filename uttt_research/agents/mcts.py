"""
Monte Carlo Tree Search Agent

A strong baseline agent that uses MCTS to select moves.
No neural network required - uses random rollouts for value estimation.
"""
import math
import random
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List, TYPE_CHECKING

from .base import Agent

if TYPE_CHECKING:
    from ..engine.state import UTTTState
    from ..engine.rules import UTTTRules


@dataclass
class MCTSNode:
    """Node in the MCTS search tree."""
    state: 'UTTTState'
    parent: Optional['MCTSNode'] = None
    move: Optional[Tuple[int, int]] = None  # Move that led to this state
    children: Dict[Tuple[int, int], 'MCTSNode'] = field(default_factory=dict)
    visits: int = 0
    value: float = 0.0  # Total value (sum of results)
    untried_moves: List[Tuple[int, int]] = field(default_factory=list)
    
    @property
    def q_value(self) -> float:
        """Average value (win rate estimate)."""
        if self.visits == 0:
            return 0.0
        return self.value / self.visits
    
    def ucb1(self, exploration: float = 1.414) -> float:
        """Upper Confidence Bound for tree policy."""
        if self.visits == 0:
            return float('inf')
        if self.parent is None:
            return self.q_value
        
        exploitation = self.q_value
        exploration_term = exploration * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration_term
    
    def best_child(self, exploration: float = 1.414) -> 'MCTSNode':
        """Select child with highest UCB1 value."""
        return max(self.children.values(), key=lambda n: n.ucb1(exploration))
    
    def is_fully_expanded(self) -> bool:
        """Check if all moves have been tried."""
        return len(self.untried_moves) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self.state.is_terminal()


class MCTSAgent(Agent):
    """
    Monte Carlo Tree Search agent.
    
    Uses UCB1 for tree policy and random rollouts for evaluation.
    Can optionally use a neural network for rollout policy and value estimation.
    """
    
    def __init__(
        self,
        num_simulations: int = 1000,
        exploration: float = 1.414,
        max_rollout_depth: int = 200,
        move_randomness: float = 0.0,
        seed: Optional[int] = None,
        name: Optional[str] = None,
        reuse_tree: bool = True,
    ):
        """
        Initialize MCTS agent.
        
        Args:
            num_simulations: Number of simulations per move
            exploration: UCB1 exploration constant (sqrt(2) is theoretically optimal)
            max_rollout_depth: Maximum depth for random rollouts
            move_randomness: Probability of selecting a random legal move
            seed: Random seed
            name: Custom name for the agent
            reuse_tree: Whether to reuse search tree across moves
        """
        self._name = name or f"MCTS({num_simulations})"
        self.num_simulations = num_simulations
        self.exploration = exploration
        self.max_rollout_depth = max_rollout_depth
        self.move_randomness = move_randomness
        self._rng = random.Random(seed)
        self._seed = seed
        self.reuse_tree = reuse_tree
        
        self._root: Optional[MCTSNode] = None
        self._rules: Optional['UTTTRules'] = None
        self._last_value: Optional[float] = None
    
    @property
    def name(self) -> str:
        return self._name
    
    def reset(self):
        """Reset agent for new game."""
        self._root = None
        self._last_value = None
        if self._seed is not None:
            self._rng = random.Random(self._seed)
    
    def select_move(
        self, 
        state: 'UTTTState', 
        rules: 'UTTTRules'
    ) -> Tuple[int, int]:
        """Select best move using MCTS."""
        self._rules = rules
        
        # Initialize or reuse root
        if self._root is None or not self.reuse_tree:
            self._root = self._create_node(state, rules)
        
        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate(self._root, rules)
        
        # Select best move (most visited child)
        if not self._root.children:
            # Edge case: no simulations expanded children
            legal_moves = rules.get_legal_moves(state)
            return self._rng.choice(legal_moves) if legal_moves else (0, 0)

        if self.move_randomness > 0 and self._rng.random() < self.move_randomness:
            legal_moves = rules.get_legal_moves(state)
            move = self._rng.choice(legal_moves) if legal_moves else (0, 0)
            if self.reuse_tree and legal_moves:
                if move in self._root.children:
                    self._root = self._root.children[move]
                    self._root.parent = None
                else:
                    next_state = rules.apply_move(state, move)
                    self._root = self._create_node(next_state, rules)
            else:
                self._root = None
            self._last_value = None
            return move

        best_child = max(
            self._root.children.values(),
            key=lambda n: n.visits
        )
        
        # Store value estimate
        self._last_value = best_child.q_value
        
        # Update root for next move if reusing tree
        if self.reuse_tree:
            self._root = best_child
            self._root.parent = None
        else:
            self._root = None
        
        return best_child.move
    
    def get_value_estimate(self, state: 'UTTTState') -> Optional[float]:
        """Get value estimate from last search."""
        return self._last_value
    
    def get_policy(self, state: 'UTTTState', rules: 'UTTTRules') -> Optional[dict]:
        """Get visit-count policy from search tree."""
        if self._root is None or not self._root.children:
            return None
        
        total_visits = sum(c.visits for c in self._root.children.values())
        if total_visits == 0:
            return None
        
        return {
            move: child.visits / total_visits
            for move, child in self._root.children.items()
        }
    
    def on_opponent_move(self, move: Tuple[int, int], state: 'UTTTState'):
        """Update root to opponent's move for tree reuse."""
        if self.reuse_tree and self._root is not None:
            if move in self._root.children:
                self._root = self._root.children[move]
                self._root.parent = None
            else:
                self._root = None
    
    def _create_node(
        self, 
        state: 'UTTTState', 
        rules: 'UTTTRules',
        parent: Optional[MCTSNode] = None,
        move: Optional[Tuple[int, int]] = None
    ) -> MCTSNode:
        """Create a new MCTS node."""
        node = MCTSNode(
            state=state,
            parent=parent,
            move=move,
            untried_moves=rules.get_legal_moves(state).copy()
        )
        self._rng.shuffle(node.untried_moves)
        return node
    
    def _simulate(self, root: MCTSNode, rules: 'UTTTRules'):
        """Run one MCTS simulation from the root."""
        node = root
        
        # Selection: traverse tree using UCB1
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child(self.exploration)
        
        # Expansion: add one new child
        if not node.is_terminal() and node.untried_moves:
            move = node.untried_moves.pop()
            new_state = rules.apply_move(node.state, move)
            child = self._create_node(new_state, rules, parent=node, move=move)
            node.children[move] = child
            node = child
        
        # Rollout: random playout to terminal state
        result = self._rollout(node.state, rules)
        
        # Backpropagation: update values up the tree
        self._backpropagate(node, result)
    
    def _rollout(self, state: 'UTTTState', rules: 'UTTTRules') -> float:
        """
        Random playout from state to terminal.
        
        Returns result from perspective of player who just moved
        (i.e., the opponent of state.current_player).
        """
        rollout_state = state.copy()
        depth = 0
        
        while not rollout_state.is_terminal() and depth < self.max_rollout_depth:
            moves = rules.get_legal_moves(rollout_state)
            if not moves:
                break
            move = self._rng.choice(moves)
            rollout_state = rules.apply_move(rollout_state, move)
            depth += 1
        
        # Get result from perspective of the player who made the move
        # that led to this state (opponent of current player at start)
        player_who_moved = state.current_player.opponent()
        result = rules.get_result(rollout_state, player_who_moved)
        
        return result if result is not None else 0.5
    
    def _backpropagate(self, node: MCTSNode, result: float):
        """Backpropagate result up the tree."""
        while node is not None:
            node.visits += 1
            # Result is from perspective of player who moved to this node
            # So we need to flip for parent (who is the opponent)
            node.value += result
            result = 1.0 - result  # Flip perspective
            node = node.parent


class MCTSAgentWithPrior(MCTSAgent):
    """
    MCTS agent that can use a neural network for prior policy and value estimation.
    
    This is used in AlphaZero-style training where the network guides search.
    """
    
    def __init__(
        self,
        policy_value_fn=None,  # Callable[[state], (policy_dict, value)]
        num_simulations: int = 800,
        exploration: float = 1.414,
        dirichlet_alpha: float = 0.3,
        dirichlet_weight: float = 0.25,
        **kwargs
    ):
        """
        Initialize MCTS with neural network priors.
        
        Args:
            policy_value_fn: Function that takes state and returns (policy, value)
            dirichlet_alpha: Dirichlet noise parameter for root exploration
            dirichlet_weight: Weight of Dirichlet noise at root
        """
        super().__init__(num_simulations=num_simulations, exploration=exploration, **kwargs)
        self.policy_value_fn = policy_value_fn
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
    
    def _rollout(self, state: 'UTTTState', rules: 'UTTTRules') -> float:
        """Use neural network value estimate instead of random rollout."""
        if self.policy_value_fn is not None:
            _, value = self.policy_value_fn(state)
            # Convert from current player perspective to perspective of
            # player who moved to this state
            return 1.0 - value
        return super()._rollout(state, rules)
