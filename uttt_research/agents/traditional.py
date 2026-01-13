"""
Traditional Neural Network Agents.

Provides a simple multilayer perceptron (MLP) policy/value network and
agents that use it directly or with MCTS for testing purposes.
"""
from typing import Tuple, Optional, Dict, List, TYPE_CHECKING
import numpy as np

from .base import Agent
from .mcts import MCTSAgent

if TYPE_CHECKING:
    from ..engine.state import UTTTState
    from ..engine.rules import UTTTRules

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


class TraditionalUTTTNet:
    """
    Simple MLP-based policy/value network for UTTT.

    Input: flattened 9x9x4 feature planes.
    Outputs: policy logits over 81 cells and a scalar value in [-1, 1].
    """

    def __init__(
        self,
        hidden_sizes: Tuple[int, int] = (256, 128),
        device: str = "cpu",
    ):
        _import_torch()
        self.device = device
        self.hidden_sizes = hidden_sizes
        self.model = self._build_network(hidden_sizes)
        self.model.to(device)
        self.model.eval()

    def _build_network(self, hidden_sizes: Tuple[int, int]):
        input_size = 4 * 9 * 9
        hidden1, hidden2 = hidden_sizes

        class TraditionalNetwork(nn.Module):
            def __init__(self, input_size, hidden1, hidden2):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden1)
                self.fc2 = nn.Linear(hidden1, hidden2)
                self.policy_head = nn.Linear(hidden2, 81)
                self.value_head = nn.Linear(hidden2, 1)

            def forward(self, x):
                out = F.relu(self.fc1(x))
                out = F.relu(self.fc2(out))
                policy = self.policy_head(out)
                value = torch.tanh(self.value_head(out))
                return policy, value

        return TraditionalNetwork(input_size, hidden1, hidden2)

    def predict(
        self,
        state: 'UTTTState',
        legal_moves: List[Tuple[int, int]],
    ) -> Tuple[Dict[Tuple[int, int], float], float]:
        """Return (policy, win_prob) for the current player."""
        features = self._state_to_features(state)
        x = torch.FloatTensor(features).view(1, -1).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.model(x)

        policy_logits = policy_logits.cpu().numpy()[0]
        value = value.cpu().item()

        policy_dict = self._logits_to_policy(policy_logits, legal_moves)
        win_prob = (value + 1) / 2

        return policy_dict, win_prob

    def _state_to_features(self, state: 'UTTTState') -> np.ndarray:
        from ..engine.state import MacroBoardStatus

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

        if state.target_macro is not None:
            macro_row, macro_col = divmod(state.target_macro, 3)
            features[
                3,
                macro_row * 3:(macro_row + 1) * 3,
                macro_col * 3:(macro_col + 1) * 3,
            ] = 1
        else:
            for macro_idx in range(9):
                if state.get_macro_status(macro_idx) == MacroBoardStatus.ONGOING:
                    macro_row, macro_col = divmod(macro_idx, 3)
                    features[
                        3,
                        macro_row * 3:(macro_row + 1) * 3,
                        macro_col * 3:(macro_col + 1) * 3,
                    ] = 1

        return features

    def _logits_to_policy(
        self,
        logits: np.ndarray,
        legal_moves: List[Tuple[int, int]],
    ) -> Dict[Tuple[int, int], float]:
        mask = np.full(81, -1e9)
        for macro_idx, cell_idx in legal_moves:
            if macro_idx >= 0:
                macro_row, macro_col = divmod(macro_idx, 3)
                cell_row, cell_col = divmod(cell_idx, 3)
                flat_idx = (macro_row * 3 + cell_row) * 9 + (macro_col * 3 + cell_col)
                mask[flat_idx] = 0

        masked_logits = logits + mask
        exp_logits = np.exp(masked_logits - np.max(masked_logits))
        probs = exp_logits / np.sum(exp_logits)

        policy = {}
        for macro_idx, cell_idx in legal_moves:
            if macro_idx >= 0:
                macro_row, macro_col = divmod(macro_idx, 3)
                cell_row, cell_col = divmod(cell_idx, 3)
                flat_idx = (macro_row * 3 + cell_row) * 9 + (macro_col * 3 + cell_col)
                policy[(macro_idx, cell_idx)] = probs[flat_idx]
            else:
                policy[(macro_idx, cell_idx)] = 0.01

        total = sum(policy.values())
        if total > 0:
            policy = {k: v / total for k, v in policy.items()}

        return policy

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()


class TraditionalNeuralAgent(Agent):
    """Greedy or temperature-sampled agent using the TraditionalUTTTNet."""

    def __init__(
        self,
        network: Optional[TraditionalUTTTNet] = None,
        temperature: float = 0.0,
        name: Optional[str] = None,
    ):
        _import_torch()
        self._name = name or "TraditionalNet"
        self.network = network or TraditionalUTTTNet()
        self.temperature = temperature
        self._last_value: Optional[float] = None
        self._last_policy: Optional[Dict] = None

    @property
    def name(self) -> str:
        return self._name

    def select_move(
        self,
        state: 'UTTTState',
        rules: 'UTTTRules',
    ) -> Tuple[int, int]:
        legal_moves = rules.get_legal_moves(state)
        if not legal_moves:
            raise ValueError("No legal moves available")

        policy, value = self.network.predict(state, legal_moves)
        self._last_value = value
        self._last_policy = policy

        if self.temperature == 0:
            return max(policy.items(), key=lambda x: x[1])[0]

        moves = list(policy.keys())
        probs = np.array([policy[m] for m in moves])
        probs = np.power(probs, 1.0 / self.temperature)
        probs = probs / np.sum(probs)
        idx = np.random.choice(len(moves), p=probs)
        return moves[idx]

    def get_value_estimate(self, state: 'UTTTState') -> Optional[float]:
        return self._last_value

    def get_policy(self, state: 'UTTTState', rules: 'UTTTRules') -> Optional[dict]:
        return self._last_policy

    def reset(self):
        self._last_value = None
        self._last_policy = None


class TraditionalNeuralMCTSAgent(MCTSAgent):
    """MCTS agent using a TraditionalUTTTNet for value rollouts."""

    def __init__(
        self,
        network: Optional[TraditionalUTTTNet] = None,
        num_simulations: int = 200,
        exploration: float = 1.414,
        name: Optional[str] = None,
        **kwargs,
    ):
        self.network = network or TraditionalUTTTNet()
        super().__init__(
            num_simulations=num_simulations,
            exploration=exploration,
            name=name or f"TraditionalNetMCTS({num_simulations})",
            **kwargs,
        )

    def _rollout(self, state: 'UTTTState', rules: 'UTTTRules') -> float:
        legal_moves = rules.get_legal_moves(state)
        if not legal_moves:
            return 0.5

        _, win_prob = self.network.predict(state, legal_moves)
        return 1.0 - win_prob
