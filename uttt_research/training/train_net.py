"""
Neural Network Training

Training loop for the AlphaZero-lite neural network.
"""
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..engine.state import UTTTState
    from ..agents.neural import UTTTNet

# Lazy torch import
torch = None
nn = None


def _import_torch():
    global torch, nn
    if torch is None:
        import torch as _torch
        import torch.nn as _nn
        torch = _torch
        nn = _nn


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    epochs_per_iteration: int = 10
    min_examples: int = 1000
    checkpoint_interval: int = 100
    device: str = "cpu"


class Trainer:
    """
    Trainer for the UTTT neural network.
    
    Implements training loop with policy and value losses.
    """
    
    def __init__(self, network: 'UTTTNet', config: Optional[TrainingConfig] = None):
        """
        Initialize trainer.
        
        Args:
            network: Neural network to train
            config: Training configuration
        """
        _import_torch()
        
        self.network = network
        self.config = config or TrainingConfig()
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            network.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Training stats
        self.train_step = 0
        self.losses = []
    
    def train_on_batch(
        self, 
        examples: List[Tuple['UTTTState', Dict, float]]
    ) -> Dict[str, float]:
        """
        Train on a batch of examples.
        
        Args:
            examples: List of (state, policy_dict, value) tuples
            
        Returns:
            Dict with loss values
        """
        if len(examples) == 0:
            return {'total_loss': 0, 'policy_loss': 0, 'value_loss': 0}
        
        self.network.model.train()
        
        # Prepare batch
        states, policies, values = self._prepare_batch(examples)
        
        # Forward pass
        policy_logits, value_preds = self.network.model(states)
        
        # Policy loss (cross-entropy)
        policy_loss = -torch.sum(policies * torch.log_softmax(policy_logits, dim=1)) / len(examples)
        
        # Value loss (MSE)
        value_loss = torch.mean((value_preds.squeeze() - values) ** 2)
        
        # Total loss
        total_loss = (
            self.config.policy_loss_weight * policy_loss + 
            self.config.value_loss_weight * value_loss
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        self.train_step += 1
        
        losses = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
        self.losses.append(losses)
        
        self.network.model.eval()
        
        return losses
    
    def train_epoch(
        self, 
        examples: List[Tuple['UTTTState', Dict, float]]
    ) -> Dict[str, float]:
        """
        Train for one epoch over all examples.
        
        Args:
            examples: All training examples
            
        Returns:
            Dict with average loss values
        """
        # Shuffle examples
        indices = np.random.permutation(len(examples))
        
        total_losses = {'total_loss': 0, 'policy_loss': 0, 'value_loss': 0}
        num_batches = 0
        
        for i in range(0, len(indices), self.config.batch_size):
            batch_indices = indices[i:i + self.config.batch_size]
            batch = [examples[j] for j in batch_indices]
            
            losses = self.train_on_batch(batch)
            
            for k in total_losses:
                total_losses[k] += losses[k]
            num_batches += 1
        
        # Average losses
        for k in total_losses:
            total_losses[k] /= max(num_batches, 1)
        
        return total_losses
    
    def _prepare_batch(
        self, 
        examples: List[Tuple['UTTTState', Dict, float]]
    ) -> Tuple:
        """Convert examples to tensors."""
        from ..engine.state import Player
        
        batch_size = len(examples)
        
        # Feature tensors
        state_features = np.zeros((batch_size, 4, 9, 9), dtype=np.float32)
        policy_targets = np.zeros((batch_size, 81), dtype=np.float32)
        value_targets = np.zeros(batch_size, dtype=np.float32)
        
        for i, (state, policy, value) in enumerate(examples):
            # State features
            state_features[i] = self._state_to_features(state)
            
            # Policy target (convert to flat 81-vector)
            for (macro_idx, cell_idx), prob in policy.items():
                if macro_idx >= 0:
                    macro_row, macro_col = divmod(macro_idx, 3)
                    cell_row, cell_col = divmod(cell_idx, 3)
                    flat_idx = (macro_row * 3 + cell_row) * 9 + (macro_col * 3 + cell_col)
                    policy_targets[i, flat_idx] = prob
            
            # Value target (convert from [0, 1] to [-1, 1])
            value_targets[i] = value * 2 - 1
        
        # Convert to tensors
        device = self.config.device
        states = torch.FloatTensor(state_features).to(device)
        policies = torch.FloatTensor(policy_targets).to(device)
        values = torch.FloatTensor(value_targets).to(device)
        
        return states, policies, values
    
    def _state_to_features(self, state: 'UTTTState') -> np.ndarray:
        """Convert game state to feature planes."""
        from ..engine.state import Player, MacroBoardStatus
        
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
            for macro_idx in range(9):
                if state.get_macro_status(macro_idx) == MacroBoardStatus.ONGOING:
                    macro_row, macro_col = divmod(macro_idx, 3)
                    features[3, macro_row*3:(macro_row+1)*3, macro_col*3:(macro_col+1)*3] = 1
        
        return features
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        torch.save({
            'model_state_dict': self.network.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_step': self.train_step,
            'losses': self.losses,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.network.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_step = checkpoint['train_step']
        self.losses = checkpoint['losses']


class TrainingPipeline:
    """
    Complete training pipeline combining self-play and network training.
    """
    
    def __init__(
        self,
        network: 'UTTTNet',
        rules,
        config: Optional[TrainingConfig] = None,
        num_self_play_games: int = 100,
        num_mcts_simulations: int = 100,
    ):
        """
        Initialize training pipeline.
        
        Args:
            network: Neural network to train
            rules: Game rules to use
            config: Training configuration
            num_self_play_games: Games per iteration
            num_mcts_simulations: MCTS simulations per move
        """
        from .self_play import ReplayBuffer
        from ..agents.mcts import MCTSAgentWithPrior
        
        self.network = network
        self.rules = rules
        self.config = config or TrainingConfig()
        self.trainer = Trainer(network, self.config)
        
        self.num_self_play_games = num_self_play_games
        self.num_mcts_simulations = num_mcts_simulations
        
        self.replay_buffer = ReplayBuffer()
        self.iteration = 0
    
    def run_iteration(self, verbose: bool = True) -> Dict[str, float]:
        """
        Run one iteration of self-play + training.
        
        Returns:
            Dict with training statistics
        """
        from .self_play import SelfPlayWorker, generate_self_play_games
        from ..agents.mcts import MCTSAgentWithPrior
        
        self.iteration += 1
        
        if verbose:
            print(f"\n=== Iteration {self.iteration} ===")
        
        # Create MCTS agent with current network
        def policy_value_fn(state):
            legal_moves = self.rules.get_legal_moves(state)
            return self.network.predict(state, legal_moves)
        
        agent = MCTSAgentWithPrior(
            policy_value_fn=policy_value_fn,
            num_simulations=self.num_mcts_simulations,
        )
        
        # Generate self-play games
        if verbose:
            print(f"Generating {self.num_self_play_games} self-play games...")
        
        def progress(done, total):
            if verbose and done % 10 == 0:
                print(f"  {done}/{total} games")
        
        records = generate_self_play_games(
            agent, self.rules, self.num_self_play_games,
            progress_callback=progress
        )
        
        # Add to replay buffer
        self.replay_buffer.add_games(records)
        
        if verbose:
            print(f"Replay buffer size: {len(self.replay_buffer)}")
        
        # Train on replay buffer
        if len(self.replay_buffer) >= self.config.min_examples:
            if verbose:
                print(f"Training for {self.config.epochs_per_iteration} epochs...")
            
            examples = self.replay_buffer.examples
            for epoch in range(self.config.epochs_per_iteration):
                losses = self.trainer.train_epoch(examples)
                if verbose:
                    print(f"  Epoch {epoch+1}: loss={losses['total_loss']:.4f} "
                          f"(policy={losses['policy_loss']:.4f}, value={losses['value_loss']:.4f})")
        else:
            if verbose:
                print(f"Not enough examples ({len(self.replay_buffer)} < {self.config.min_examples})")
            losses = {'total_loss': 0, 'policy_loss': 0, 'value_loss': 0}
        
        return losses
