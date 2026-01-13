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
from typing import List, Dict, Optional, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from uttt_research.engine.rules import UTTTRules
from uttt_research.engine.variants import StandardRules, RandomizedOpeningRules, SymmetricOpeningRules
from uttt_research.engine.variants.balanced import (
    PieRuleUTTT, OpenBoardRules, WonBoardPlayRules, BalancedRules
)
from uttt_research.engine.metrics import VariantStatistics, MetricsCollector, compare_variants
from uttt_research.agents import MCTSAgent
from uttt_research.experiments.game_logging import (
    StudyGameLogger,
    build_move_sequence,
    format_board_rows,
    format_macro_board_rows,
)


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


def run_mcts_skill_evaluations(
    variants: List[UTTTRules],
    num_games: int = 100,
    mcts_simulations: int = 100,
    randomness_levels: Optional[List[float]] = None,
    seed: int = 42,
    game_logger: Optional[StudyGameLogger] = None
) -> Dict[str, Dict[str, VariantStatistics]]:
    """
    Run MCTS vs MCTS evaluations with varying randomness levels.
    
    This simulates players of different skill levels.
    """
    print("\n" + "=" * 60)
    print("MCTS SKILL EVALUATIONS (RANDOMNESS LEVELS)")
    print("=" * 60)
    
    if randomness_levels is None:
        randomness_levels = [0.0, 0.1, 0.25, 0.5]

    results: Dict[str, Dict[str, VariantStatistics]] = {}
    
    for randomness in randomness_levels:
        label = f"mcts_random_{randomness:.2f}"
        print(f"\nRandomness level: {randomness:.2f}")
        print("-" * 60)
        level_results = {}

        for rules in variants:
            print(f"\n--- {rules.name} ---")

            stats = VariantStatistics(variant_name=rules.name)
            agent = MCTSAgent(
                num_simulations=mcts_simulations,
                seed=seed,
                move_randomness=randomness,
                name=f"MCTS({mcts_simulations}, rand={randomness:.2f})",
            )

            def value_estimator(state):
                return agent.get_value_estimate(state) or 0.5

            for i in range(num_games):
                # Play a game
                state = rules.create_initial_state()
                agent.reset()
                collector = MetricsCollector(value_estimator)
                move_records: List[Dict[str, Any]] = []

                while not state.is_terminal():
                    legal_moves = rules.get_legal_moves(state)
                    collector.record_turn(state, legal_moves)

                    move = agent.select_move(state, rules)
                    collector.record_move(move)

                    if game_logger:
                        move_records.append({
                            "turn": len(move_records) + 1,
                            "player": str(state.current_player),
                            "macro_idx": move[0],
                            "cell_idx": move[1],
                            "target_macro": state.target_macro,
                            "legal_moves": [list(m) for m in legal_moves],
                            "legal_moves_count": len(legal_moves),
                        })

                    state = rules.apply_move(state, move)

                metrics = collector.finalize(state)
                stats.games.append(metrics)

                if game_logger:
                    winner = state.get_winner()
                    winner_label = str(winner) if winner is not None else None
                    game_id = f"{label}:{rules.name}:{i + 1}"
                    raw_record = {
                        "game_id": game_id,
                        "variant": rules.name,
                        "randomness_level": randomness,
                        "mcts_simulations": mcts_simulations,
                        "seed": seed,
                        "agent": agent.name,
                        "moves": move_records,
                        "final_state": {
                            "winner": winner_label,
                            "macro_board": state.macro_board.tolist(),
                            "board": state.board.tolist(),
                            "macro_board_rows": format_macro_board_rows(state.macro_board),
                            "board_rows": format_board_rows(state.board),
                        },
                        "metrics": {
                            "winner": winner_label,
                            "game_length": metrics.game_length,
                            "move_freedoms": metrics.move_freedoms,
                            "macro_board_moves": metrics.macro_board_moves,
                            "value_history": metrics.value_history,
                            "constraint_runs": metrics.constraint_runs,
                        },
                    }
                    llm_moves = [
                        {
                            "turn": move["turn"],
                            "player": move["player"],
                            "move": f"{move['macro_idx']}/{move['cell_idx']}",
                            "target_macro": move["target_macro"],
                            "legal_moves_count": move["legal_moves_count"],
                        }
                        for move in move_records
                    ]
                    llm_record = {
                        "game_id": game_id,
                        "variant": rules.name,
                        "randomness_level": randomness,
                        "winner": winner_label or "Draw",
                        "move_count": metrics.game_length,
                        "move_sequence": build_move_sequence(move_records),
                        "moves": llm_moves,
                        "macro_board_rows": format_macro_board_rows(state.macro_board),
                        "board_rows": format_board_rows(state.board),
                    }
                    game_logger.record_game(raw_record, llm_record)

                if (i + 1) % 10 == 0:
                    print(f"  {i+1}/{num_games} games")

            print(stats)
            level_results[rules.name] = stats

        results[label] = level_results
    
    return results


def run_variant_study(
    num_skill_games: int = 100,
    mcts_simulations: int = 100,
    randomness_levels: Optional[List[float]] = None,
    output_dir: Optional[str] = None,
    variants: Optional[List[str]] = None,
    seed: int = 42,
    save_games: bool = False
) -> Dict:
    """
    Run a complete variant study.
    
    Args:
        num_skill_games: Number of games per randomness level per variant
        mcts_simulations: MCTS simulations per move
        randomness_levels: MCTS move randomness levels to evaluate
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
    study_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    if randomness_levels is None:
        randomness_levels = [0.0, 0.1, 0.25, 0.5]
    
    # Run evaluations
    game_logger = None
    if output_dir and save_games:
        config = {
            "timestamp": datetime.now().isoformat(),
            "num_skill_games": num_skill_games,
            "mcts_simulations": mcts_simulations,
            "randomness_levels": randomness_levels,
            "seed": seed,
            "variants": [v.name for v in all_variants],
        }
        game_logger = StudyGameLogger(Path(output_dir), study_id, config, write_llm=True)

    try:
        skill_results = run_mcts_skill_evaluations(
            all_variants,
            num_games=num_skill_games,
            mcts_simulations=mcts_simulations,
            randomness_levels=randomness_levels,
            seed=seed,
            game_logger=game_logger,
        )
    finally:
        if game_logger:
            game_logger.close()
    
    # Generate comparison report
    for label, level_results in skill_results.items():
        print("\n" + "=" * 60)
        print(f"COMPARISON REPORT ({label})")
        print("=" * 60)
        print(compare_variants(list(level_results.values())))
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")
    
    # Prepare results
    results = {
        'timestamp': datetime.now().isoformat(),
        'study_id': study_id,
        'config': {
            'num_skill_games': num_skill_games,
            'mcts_simulations': mcts_simulations,
            'randomness_levels': randomness_levels,
            'seed': seed,
            'save_games': save_games,
        },
        'skill_evaluations': {
            label: {name: stats.summary() for name, stats in level_results.items()}
            for label, level_results in skill_results.items()
        },
    }
    
    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / f"study_{study_id}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run UTTT variant comparison study")
    parser.add_argument('--skill-games', type=int, default=100,
                       help='Number of games per randomness level per variant')
    parser.add_argument('--mcts-sims', type=int, default=100,
                       help='MCTS simulations per move')
    parser.add_argument('--randomness-levels', type=float, nargs='+', default=None,
                       help='Move randomness levels for MCTS skill evaluation')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--variants', type=str, nargs='+', default=None,
                       help='Specific variants to test')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with reduced games')
    parser.add_argument('--save-games', action='store_true',
                       help='Save full per-game logs to the output directory')
    
    args = parser.parse_args()
    
    if args.quick:
        args.skill_games = 10
        args.mcts_sims = 50
    
    run_variant_study(
        num_skill_games=args.skill_games,
        mcts_simulations=args.mcts_sims,
        randomness_levels=args.randomness_levels,
        output_dir=args.output,
        variants=args.variants,
        seed=args.seed,
        save_games=args.save_games,
    )


if __name__ == '__main__':
    main()
