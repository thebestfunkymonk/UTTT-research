#!/usr/bin/env python3
"""
Test script to verify the UTTT research harness works correctly.
"""
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_basic_game():
    """Test a basic game with standard rules."""
    print("=" * 50)
    print("Test 1: Basic Game with Standard Rules")
    print("=" * 50)
    
    from uttt_research.engine.variants import StandardRules
    from uttt_research.agents import RandomAgent
    
    rules = StandardRules()
    agent = RandomAgent(seed=42)
    
    state = rules.create_initial_state()
    print(f"Initial state:\n{state}\n")
    
    move_count = 0
    while not state.is_terminal() and move_count < 100:
        move = agent.select_move(state, rules)
        state = rules.apply_move(state, move)
        move_count += 1
    
    print(f"Game ended after {move_count} moves")
    print(f"Winner: {state.get_winner()}")
    print("✓ Basic game test passed!\n")
    return True


def test_mcts_agent():
    """Test MCTS agent."""
    print("=" * 50)
    print("Test 2: MCTS Agent")
    print("=" * 50)
    
    from uttt_research.engine.variants import StandardRules
    from uttt_research.agents import MCTSAgent, RandomAgent
    
    rules = StandardRules()
    mcts = MCTSAgent(num_simulations=50, seed=42)
    random_agent = RandomAgent(seed=123)
    
    state = rules.create_initial_state()
    
    # Play a few moves alternating MCTS and Random
    for i in range(10):
        if state.is_terminal():
            break
        
        if i % 2 == 0:
            move = mcts.select_move(state, rules)
            print(f"Move {i+1}: MCTS plays {move}")
        else:
            move = random_agent.select_move(state, rules)
            print(f"Move {i+1}: Random plays {move}")
            mcts.on_opponent_move(move, state)
        
        state = rules.apply_move(state, move)
    
    print("✓ MCTS agent test passed!\n")
    return True


def test_metrics_collection():
    """Test metrics collection during a game."""
    print("=" * 50)
    print("Test 3: Metrics Collection")
    print("=" * 50)
    
    from uttt_research.engine.variants import StandardRules
    from uttt_research.engine.metrics import MetricsCollector, VariantStatistics
    from uttt_research.agents import RandomAgent
    
    rules = StandardRules()
    agent = RandomAgent(seed=42)
    collector = MetricsCollector()
    
    state = rules.create_initial_state()
    
    while not state.is_terminal():
        legal_moves = rules.get_legal_moves(state)
        collector.record_turn(state, legal_moves)
        
        move = agent.select_move(state, rules)
        collector.record_move(move)
        
        state = rules.apply_move(state, move)
    
    metrics = collector.finalize(state)
    
    print(f"Game length: {metrics.game_length}")
    print(f"Avg move freedom: {metrics.avg_move_freedom:.1f}")
    print(f"Board utilization: {metrics.board_utilization_entropy:.2f}")
    print(f"Winner: {metrics.winner}")
    print("✓ Metrics collection test passed!\n")
    return True


def test_variant_comparison():
    """Test comparing multiple variants."""
    print("=" * 50)
    print("Test 4: Variant Comparison (10 games each)")
    print("=" * 50)
    
    from uttt_research.engine.variants import StandardRules, RandomizedOpeningRules
    from uttt_research.engine.metrics import VariantStatistics, MetricsCollector
    from uttt_research.agents import RandomAgent
    
    variants = [StandardRules(), RandomizedOpeningRules(seed=42)]
    agent = RandomAgent(seed=42)
    
    for rules in variants:
        stats = VariantStatistics(variant_name=rules.name)
        
        for _ in range(10):
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
        
        print(f"\n{rules.name}:")
        summary = stats.summary()
        print(f"  X win rate: {summary['x_win_rate']*100:.0f}%")
        print(f"  O win rate: {summary['o_win_rate']*100:.0f}%")
        print(f"  Avg game length: {summary['avg_game_length']:.1f}")
        print(f"  Avg move freedom: {summary['avg_move_freedom']:.1f}")
    
    print("\n✓ Variant comparison test passed!\n")
    return True


def test_arena():
    """Test the evaluation arena."""
    print("=" * 50)
    print("Test 5: Arena Evaluation (MCTS vs Random)")
    print("=" * 50)
    
    from uttt_research.engine.variants import StandardRules
    from uttt_research.agents import MCTSAgent, RandomAgent
    from uttt_research.training.evaluator import Arena
    
    rules = StandardRules()
    mcts = MCTSAgent(num_simulations=50, seed=42, name="MCTS-50")
    random_agent = RandomAgent(seed=123, name="Random")
    
    arena = Arena(rules)
    stats = arena.play_matches(mcts, random_agent, num_games=10, alternate=True)
    
    print(f"\nResults: {stats}")
    print("✓ Arena test passed!\n")
    return True


def test_all_variants():
    """Test that all variants can run a game."""
    print("=" * 50)
    print("Test 6: All Variants Basic Play")
    print("=" * 50)
    
    from uttt_research.engine.variants import (
        StandardRules, RandomizedOpeningRules, SymmetricOpeningRules,
        PieRuleUTTT, OpenBoardRules, WonBoardPlayRules, BalancedRules
    )
    from uttt_research.agents import RandomAgent
    
    variants = [
        StandardRules(),
        RandomizedOpeningRules(seed=42),
        SymmetricOpeningRules(seed=42),
        PieRuleUTTT(),
        OpenBoardRules(),
        WonBoardPlayRules(),
        BalancedRules(),
    ]
    
    agent = RandomAgent(seed=42)
    
    for rules in variants:
        state = rules.create_initial_state()
        move_count = 0
        
        while not state.is_terminal() and move_count < 100:
            move = agent.select_move(state, rules)
            state = rules.apply_move(state, move)
            move_count += 1
        
        winner = state.get_winner()
        status = f"Winner: {winner}" if winner else "Draw"
        print(f"  {rules.name}: {move_count} moves, {status}")
    
    print("✓ All variants test passed!\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("UTTT RESEARCH HARNESS - TEST SUITE")
    print("=" * 60 + "\n")
    
    tests = [
        test_basic_game,
        test_mcts_agent,
        test_metrics_collection,
        test_variant_comparison,
        test_arena,
        test_all_variants,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
