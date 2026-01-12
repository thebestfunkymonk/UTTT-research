#!/usr/bin/env python3
"""
Run Study - Full Comparison of UTTT Variants

This script runs a comprehensive comparison of different UTTT rule variants,
evaluating them on fairness, game dynamics, and learning potential.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from uttt_research.engine import UTTTState, Player
from uttt_research.engine.rules import UTTTRules
from uttt_research.engine.variants import StandardRules, RandomizedOpeningRules, SymmetricOpeningRules
from uttt_research.engine.variants.balanced import (
    PieRuleUTTT, OpenBoardRules, WonBoardPlayRules, KillMoveRules, BalancedRules
)
from uttt_research.engine.metrics import VariantStatistics, MetricsCollector, compare_variants
from uttt_research.agents import RandomAgent, MCTSAgent
from uttt_research.training.evaluator import Arena, quick_evaluate


def get_all_variants() -> List[UTTTRules]:
    """Get all available rule variants."""
    return [
        StandardRules(),
        RandomizedOpeningRules(seed=42),
        SymmetricOpeningRules(seed=42),
        PieRuleUTTT(),
        OpenBoardRules(),
        WonBoardPlayRules(),
        BalancedRules(),
    ]


def run_random_baseline(
    variants: List[UTTTRules],
    num_games: int = 1000,
    seed: int = 42
) -> Dict[str, VariantStatistics]:
    """
    Run random vs random baseline on all variants.
    
    This gives us basic statistics without computational overhead.
    """
    print("\n" + "=" * 60)
    print("RANDOM BASELINE EVALUATION")
    print("=" * 60)
    
    results = {}
    
    for rules in variants:
        print(f"\n--- {rules.name} ---")
        
        stats = VariantStatistics(variant_name=rules.name)
        agent = RandomAgent(seed=seed)
        
        for i in range(num_games):
            # Play a game
            state = rules.create_initial_state()
            collector = MetricsCollector()
            
            while not state.is_terminal():
                legal_moves = rules.get_legal_moves(state)
                collector.record_turn(state, legal_moves)
                
                move = agent.select_move(state, rules)
                collector.record_move(move)
                
                state = rules.apply_move(state, move)
            
            metrics = collector.finalize(state)
            stats.games.append(metrics)
            
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{num_games} games")
        
        print(stats)
        results[rules.name] = stats
    
    return results


def run_mcts_evaluation(
    variants: List[UTTTRules],
    num_games: int = 100,
    mcts_simulations: int = 100,
    seed: int = 42
) -> Dict[str, VariantStatistics]:
    """
    Run MCTS vs MCTS evaluation on all variants.
    
    This gives us more accurate win rates and value-based metrics.
    """
    print("\n" + "=" * 60)
    print("MCTS EVALUATION")
    print(f"({mcts_simulations} simulations per move)")
    print("=" * 60)
    
    results = {}
    
    for rules in variants:
        print(f"\n--- {rules.name} ---")
        
        stats = VariantStatistics(variant_name=rules.name)
        agent = MCTSAgent(num_simulations=mcts_simulations, seed=seed)
        
        def value_estimator(state):
            return agent.get_value_estimate(state) or 0.5
        
        for i in range(num_games):
            # Play a game
            state = rules.create_initial_state()
            agent.reset()
            collector = MetricsCollector(value_estimator)
            
            while not state.is_terminal():
                legal_moves = rules.get_legal_moves(state)
                collector.record_turn(state, legal_moves)
                
                move = agent.select_move(state, rules)
                collector.record_move(move)
                
                state = rules.apply_move(state, move)
            
            metrics = collector.finalize(state)
            stats.games.append(metrics)
            
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{num_games} games")
        
        print(stats)
        results[rules.name] = stats
    
    return results


def run_variant_study(
    num_random_games: int = 1000,
    num_mcts_games: int = 100,
    mcts_simulations: int = 100,
    output_dir: Optional[str] = None,
    variants: Optional[List[str]] = None,
    seed: int = 42
) -> Dict:
    """
    Run a complete variant study.
    
    Args:
        num_random_games: Number of random vs random games per variant
        num_mcts_games: Number of MCTS vs MCTS games per variant
        mcts_simulations: MCTS simulations per move
        output_dir: Directory to save results
        variants: List of variant names to test (None = all)
        seed: Random seed
        
    Returns:
        Dict with all results
    """
    print("=" * 60)
    print("UTTT VARIANT STUDY")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Get variants
    all_variants = get_all_variants()
    if variants:
        all_variants = [v for v in all_variants if v.name in variants]
    
    print(f"\nVariants to test: {[v.name for v in all_variants]}")
    
    start_time = time.time()
    
    # Run evaluations
    random_results = run_random_baseline(all_variants, num_random_games, seed)
    
    if num_mcts_games > 0:
        mcts_results = run_mcts_evaluation(all_variants, num_mcts_games, mcts_simulations, seed)
    else:
        mcts_results = {}
    
    # Generate comparison report
    print("\n" + "=" * 60)
    print("COMPARISON REPORT (Random Baseline)")
    print("=" * 60)
    print(compare_variants(list(random_results.values())))
    
    if mcts_results:
        print("\n" + "=" * 60)
        print("COMPARISON REPORT (MCTS)")
        print("=" * 60)
        print(compare_variants(list(mcts_results.values())))
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")
    
    # Prepare results
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'num_random_games': num_random_games,
            'num_mcts_games': num_mcts_games,
            'mcts_simulations': mcts_simulations,
            'seed': seed,
        },
        'random_baseline': {
            name: stats.summary() for name, stats in random_results.items()
        },
        'mcts': {
            name: stats.summary() for name, stats in mcts_results.items()
        } if mcts_results else {},
    }
    
    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run UTTT variant comparison study")
    parser.add_argument('--random-games', type=int, default=1000,
                       help='Number of random vs random games per variant')
    parser.add_argument('--mcts-games', type=int, default=100,
                       help='Number of MCTS vs MCTS games per variant')
    parser.add_argument('--mcts-sims', type=int, default=100,
                       help='MCTS simulations per move')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--variants', type=str, nargs='+', default=None,
                       help='Specific variants to test')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with reduced games')
    
    args = parser.parse_args()
    
    if args.quick:
        args.random_games = 100
        args.mcts_games = 10
        args.mcts_sims = 50
    
    run_variant_study(
        num_random_games=args.random_games,
        num_mcts_games=args.mcts_games,
        mcts_simulations=args.mcts_sims,
        output_dir=args.output,
        variants=args.variants,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
