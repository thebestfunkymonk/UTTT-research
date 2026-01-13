# UTTT Research Harness

A modular research framework for studying Ultimate Tic-Tac-Toe (UTTT) variants and training AI agents.

## Overview

This project provides:
- **Flexible game engine** with support for multiple rule variants
- **Agent framework** including Random, MCTS, convolutional NN, and traditional MLP agents
- **Training infrastructure** for AlphaZero-style self-play learning
- **Evaluation tools** for comparing variants and agents
- **Metrics collection** for analyzing game dynamics

## Installation

```bash
# Clone the repository
cd UTTT-research

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run a test match

```python
from uttt_research.engine.variants import StandardRules
from uttt_research.agents import RandomAgent, MCTSAgent
from uttt_research.training.evaluator import quick_evaluate

rules = StandardRules()
agent_random = RandomAgent(seed=42)
agent_mcts = MCTSAgent(num_simulations=100, seed=42)

# Evaluate MCTS vs Random
stats = quick_evaluate(agent_mcts, agent_random, rules, num_games=100)
print(stats)
```

### Run the variant study

```bash
# Quick test
python -m uttt_research.experiments.run_study --quick

# Full study
python -m uttt_research.experiments.run_study --skill-games 1000 --mcts-sims 100
```

### Run the test harness

```bash
python -m uttt_research.test_harness
```

### Quick NN-based evaluation (one-off command)

Use the neural agent in a basic evaluation run (requires PyTorch installed).

```bash
python - <<'PY'
from uttt_research.engine.variants import StandardRules
from uttt_research.agents import MCTSAgent
from uttt_research.agents.neural import NeuralNetworkAgent
from uttt_research.training.evaluator import quick_evaluate

rules = StandardRules()
nn_agent = NeuralNetworkAgent()
mcts_agent = MCTSAgent(num_simulations=100, seed=42)

quick_evaluate(nn_agent, mcts_agent, rules, num_games=20)
PY
```

## Project Structure

```
uttt_research/
├── engine/
│   ├── state.py        # Board representation
│   ├── rules.py        # Abstract rule interface
│   ├── metrics.py      # Game metrics collection
│   └── variants/
│       ├── standard.py      # Standard UTTT
│       ├── balanced.py      # Pie Rule, Open Board, etc.
│       └── randomized_opening.py  # Randomized openings
├── agents/
│   ├── base.py         # Agent interface
│   ├── random.py       # Random baseline
│   ├── mcts.py         # Monte Carlo Tree Search
│   ├── neural.py       # Convolutional policy/value network agents
│   └── traditional.py  # MLP policy/value network agents
├── training/
│   ├── self_play.py    # Self-play data generation
│   ├── train_net.py    # Neural network training
│   └── evaluator.py    # Agent evaluation arena
└── experiments/
    └── run_study.py    # Variant comparison study
```

## Rule Variants

| Variant | Description |
|---------|-------------|
| **Standard** | Classic UTTT rules - can only play in ONGOING boards |
| **Pie Rule** | P2 can swap sides after P1's first move |
| **Open Board** | When sent to WON board, can play in ANY ONGOING or WON board |
| **Won-Board Play** | Can play in WON boards (if not FULL) - treated like ONGOING when targeted |
| **Kill-Move** | Special constraints after winning a macro-board |
| **Balanced** | Combines Pie Rule + Open Board for fairness |
| **Randomized Opening** | Pre-placed moves to disrupt forced wins |
| **Symmetric Opening** | Symmetric placement for balance |

*Note: ONGOING = not won and not full; WON = won but not full; FULL = no empty cells*

## Metrics

The framework evaluates variants on:

- **Fairness**: P1 vs P2 win rates (target: 50/50)
- **Decisiveness**: Draw rate (target: 0%)
- **Move Freedom**: Average legal moves per turn
- **Board Utilization**: Distribution of play across macro-boards
- **Drama**: Frequency of lead changes
- **Constraint Intensity**: Length of forced-move sequences
- **Game Length**: Average moves to terminal state

## Training Neural Networks

```python
from uttt_research.agents.neural import UTTTNet
from uttt_research.training.train_net import TrainingPipeline, TrainingConfig
from uttt_research.engine.variants import StandardRules

# Create network and pipeline
network = UTTTNet(num_residual_blocks=4, num_filters=64)
rules = StandardRules()
config = TrainingConfig(batch_size=256, learning_rate=0.001)

pipeline = TrainingPipeline(
    network, rules, config,
    num_self_play_games=100,
    num_mcts_simulations=100
)

# Run training iterations
for i in range(10):
    losses = pipeline.run_iteration()
```

## CLI Commands and Flags

### Variant study (`uttt_research.experiments.run_study`)

Run a comparative study across UTTT variants using MCTS agents.

```bash
python -m uttt_research.experiments.run_study [options]
```

**Options**
- `--skill-games <int>`: Games per randomness level per variant (default: 100).
- `--mcts-sims <int>`: MCTS simulations per move (default: 100).
- `--randomness-levels <float> [<float> ...]`: Randomness levels for MCTS skill evaluation (default: `[0.0, 0.1, 0.25, 0.5]`).
- `--output <path>`: Output directory for JSON results.
- `--seed <int>`: Random seed (default: 42).
- `--variants <name> [<name> ...]`: Limit the study to specific variant names (matches `rules.name`).
- `--quick`: Shortcut for a reduced run (`--skill-games 10 --mcts-sims 50`).

**Examples**
```bash
# Fast sanity check
python -m uttt_research.experiments.run_study --quick

# Targeted run on specific variants
python -m uttt_research.experiments.run_study \
  --variants Standard Balanced \
  --skill-games 200 \
  --mcts-sims 200 \
  --randomness-levels 0.0 0.1
```

## References

- [A Practical Method for Preventing Forced Wins in Ultimate Tic-Tac-Toe](https://arxiv.org/abs/2207.06239)
- [At Most 43 Moves, At Least 29: Optimal Strategies and Bounds for Ultimate Tic-Tac-Toe](https://arxiv.org/abs/2006.02353)

## License

MIT
