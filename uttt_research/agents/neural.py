"""
Neural Network Agent

A neural network-based agent using PyTorch.
Implements an AlphaZero-lite architecture with policy and value heads.
"""
from typing import Tuple, Optional, Dict, List, TYPE_CHECKING
import numpy as np

from .base import Agent

if TYPE_CHECKING:
    from ..engine.state import UTTTState, Player
    from ..engine.rules import UTTTRules

# Lazy imports for PyTorch (not everyone needs it)
torch = None
nn = None
F = None


def _import_torch():
    """Lazily import PyTorch."""
    global torch, nn, F
    if torch is None:
        import torch as _torch
        import torch.nn as _nn
        import torch.nn.functional as _F
        torch = _torch
        nn = _nn
        F = _F


class UTTTNet:
    """
    Neural network for UTTT.
    
    Architecture:
    - Input: 9x9x4 feature planes (current player, opponent, empty, valid moves)
    - Residual tower with N blocks
    - Policy head: 9x9 output (logits for each cell)
    - Value head: Single scalar (-1 to 1)
    """
    
    def __init__(
        self,
        num_residual_blocks: int = 4,
        num_filters: int = 64,
        device: str = "cpu"
    ):
        """
        Initialize the network.
        
        Args:
            num_residual_blocks: Number of residual blocks in the tower
            num_filters: Number of convolutional filters
            device: Device to run on ("cpu", "cuda", "mps")
        """
        _import_torch()
        
        self.device = device
        self.num_filters = num_filters
        
        # Build the network
        self.model = self._build_network(num_residual_blocks, num_filters)
        self.model.to(device)
        self.model.eval()
    
    def _build_network(self, num_blocks: int, filters: int):
        """Build the neural network architecture."""
        
        class ResidualBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(channels)
                self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(channels)
            
            def forward(self, x):
                residual = x
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += residual
                return F.relu(out)
        
        class UTTTNetwork(nn.Module):
            def __init__(self, num_blocks, filters):
                super().__init__()
                
                # Input convolution
                self.input_conv = nn.Conv2d(4, filters, 3, padding=1)
                self.input_bn = nn.BatchNorm2d(filters)
                
                # Residual tower
                self.residual_blocks = nn.ModuleList([
                    ResidualBlock(filters) for _ in range(num_blocks)
                ])
                
                # Policy head
                self.policy_conv = nn.Conv2d(filters, 2, 1)
                self.policy_bn = nn.BatchNorm2d(2)
                self.policy_fc = nn.Linear(2 * 9 * 9, 81)
                
                # Value head
                self.value_conv = nn.Conv2d(filters, 1, 1)
                self.value_bn = nn.BatchNorm2d(1)
                self.value_fc1 = nn.Linear(81, 64)
                self.value_fc2 = nn.Linear(64, 1)
            
            def forward(self, x):
                # Input
                out = F.relu(self.input_bn(self.input_conv(x)))
                
                # Residual tower
                for block in self.residual_blocks:
                    out = block(out)
                
                # Policy head
                policy = F.relu(self.policy_bn(self.policy_conv(out)))
                policy = policy.view(-1, 2 * 9 * 9)
                policy = self.policy_fc(policy)
                
                # Value head
                value = F.relu(self.value_bn(self.value_conv(out)))
                value = value.view(-1, 81)
                value = F.relu(self.value_fc1(value))
                value = torch.tanh(self.value_fc2(value))
                
                return policy, value
        
        return UTTTNetwork(num_blocks, filters)
    
    def predict(self, state: 'UTTTState', legal_moves: List[Tuple[int, int]]) -> Tuple[Dict, float]:
        """
        Get policy and value predictions for a state.
        
        Args:
            state: Game state
            legal_moves: List of legal moves
            
        Returns:
            (policy_dict, value) where policy_dict maps moves to probabilities
        """
        # Convert state to tensor
        features = self._state_to_features(state)
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            policy_logits, value = self.model(x)
        
        policy_logits = policy_logits.cpu().numpy()[0]
        value = value.cpu().item()
        
        # Mask illegal moves and compute softmax
        policy_dict = self._logits_to_policy(policy_logits, legal_moves)
        
        # Convert value from [-1, 1] to [0, 1]
        win_prob = (value + 1) / 2
        
        return policy_dict, win_prob
    
    def _state_to_features(self, state: 'UTTTState') -> np.ndarray:
        """
        Convert game state to feature planes.
        
        Planes:
        0: Current player's pieces
        1: Opponent's pieces
        2: Empty cells
        3: Valid macro-boards (where moves can be made)
        """
        from ..engine.state import Player
        
        features = np.zeros((4, 9, 9), dtype=np.float32)
        
        current = state.current_player
        opponent = current.opponent()
        
        for row in range(9):
            for col in range(9):
                cell = state.board[row, col]
                if cell == current:
                    features[0, row, col] = 1
                elif cell == opponent:
                    features[1, row, col] = 1
                else:
                    features[2, row, col] = 1
        
        # Valid macro-boards
        if state.target_macro is not None:
            macro_row, macro_col = divmod(state.target_macro, 3)
            features[3, macro_row*3:(macro_row+1)*3, macro_col*3:(macro_col+1)*3] = 1
        else:
            # All undecided macro-boards are valid
            from ..engine.state import MacroBoardStatus
            for macro_idx in range(9):
                if state.get_macro_status(macro_idx) == MacroBoardStatus.ONGOING:
                    macro_row, macro_col = divmod(macro_idx, 3)
                    features[3, macro_row*3:(macro_row+1)*3, macro_col*3:(macro_col+1)*3] = 1
        
        return features
    
    def _logits_to_policy(
        self, 
        logits: np.ndarray, 
        legal_moves: List[Tuple[int, int]]
    ) -> Dict[Tuple[int, int], float]:
        """Convert logits to policy distribution over legal moves."""
        # Create mask for legal moves
        mask = np.full(81, -1e9)
        for macro_idx, cell_idx in legal_moves:
            if macro_idx >= 0:  # Skip special moves like swap
                macro_row, macro_col = divmod(macro_idx, 3)
                cell_row, cell_col = divmod(cell_idx, 3)
                flat_idx = (macro_row * 3 + cell_row) * 9 + (macro_col * 3 + cell_col)
                mask[flat_idx] = 0
        
        # Apply mask and softmax
        masked_logits = logits + mask
        exp_logits = np.exp(masked_logits - np.max(masked_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Build policy dict
        policy = {}
        for macro_idx, cell_idx in legal_moves:
            if macro_idx >= 0:
                macro_row, macro_col = divmod(macro_idx, 3)
                cell_row, cell_col = divmod(cell_idx, 3)
                flat_idx = (macro_row * 3 + cell_row) * 9 + (macro_col * 3 + cell_col)
                policy[(macro_idx, cell_idx)] = probs[flat_idx]
            else:
                # Special moves get small probability
                policy[(macro_idx, cell_idx)] = 0.01
        
        # Normalize
        total = sum(policy.values())
        if total > 0:
            policy = {k: v/total for k, v in policy.items()}
        
        return policy
    
    def save(self, path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path: str):
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()


class NeuralNetworkAgent(Agent):
    """
    Agent that uses a neural network for move selection.
    
    Can be used directly (greedy policy) or combined with MCTS.
    """
    
    def __init__(
        self,
        network: Optional[UTTTNet] = None,
        temperature: float = 0.0,
        name: Optional[str] = None
    ):
        """
        Initialize neural network agent.
        
        Args:
            network: UTTTNet instance (creates new one if None)
            temperature: Sampling temperature (0 = greedy, higher = more exploration)
            name: Custom name
        """
        _import_torch()
        
        self._name = name or "NeuralNet"
        self.network = network or UTTTNet()
        self.temperature = temperature
        self._last_value: Optional[float] = None
        self._last_policy: Optional[Dict] = None
    
    @property
    def name(self) -> str:
        return self._name
    
    def select_move(
        self, 
        state: 'UTTTState', 
        rules: 'UTTTRules'
    ) -> Tuple[int, int]:
        """Select move using neural network policy."""
        legal_moves = rules.get_legal_moves(state)
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        policy, value = self.network.predict(state, legal_moves)
        self._last_value = value
        self._last_policy = policy
        
        if self.temperature == 0:
            # Greedy selection
            return max(policy.items(), key=lambda x: x[1])[0]
        else:
            # Temperature-based sampling
            moves = list(policy.keys())
            probs = np.array([policy[m] for m in moves])
            
            # Apply temperature
            probs = np.power(probs, 1.0 / self.temperature)
            probs = probs / np.sum(probs)
            
            idx = np.random.choice(len(moves), p=probs)
            return moves[idx]
    
    def get_value_estimate(self, state: 'UTTTState') -> Optional[float]:
        """Get value estimate from last prediction."""
        return self._last_value
    
    def get_policy(self, state: 'UTTTState', rules: 'UTTTRules') -> Optional[dict]:
        """Get policy from last prediction."""
        return self._last_policy
    
    def reset(self):
        """Reset stored predictions."""
        self._last_value = None
        self._last_policy = None
