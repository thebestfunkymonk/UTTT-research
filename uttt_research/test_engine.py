#!/usr/bin/env python3
"""
Simple test to verify the UTTT engine works correctly.
"""
import random
from engine import UTTTState, Player, MacroBoardStatus
from engine.variants import StandardRules, RandomizedOpeningRules, SymmetricOpeningRules


def test_standard_rules():
    """Test standard UTTT rules."""
    print("=" * 50)
    print("Testing Standard Rules")
    print("=" * 50)
    
    rules = StandardRules()
    state = rules.create_initial_state()
    
    print(f"Rule variant: {rules.name}")
    print(f"Initial state:")
    print(state)
    
    # Get legal moves
    moves = rules.get_legal_moves(state)
    print(f"\nLegal moves at start: {len(moves)}")
    
    # Play a few random moves
    rng = random.Random(42)
    for i in range(10):
        if state.is_terminal():
            print(f"\nGame ended after {i} moves!")
            break
        
        moves = rules.get_legal_moves(state)
        if not moves:
            print(f"\nNo legal moves after {i} moves!")
            break
            
        move = rng.choice(moves)
        state = rules.apply_move(state, move)
        print(f"\nMove {i+1}: Player {state.current_player.opponent()} plays {move}")
    
    print(f"\nState after moves:")
    print(state)
    
    return True


def test_randomized_opening():
    """Test randomized opening rules."""
    print("\n" + "=" * 50)
    print("Testing Randomized Opening Rules")
    print("=" * 50)
    
    # Test with specific seed for reproducibility
    rules = RandomizedOpeningRules(seed=12345)
    state = rules.create_initial_state()
    
    print(f"Rule variant: {rules.name}")
    print(f"Opening code: {rules.get_opening_code()}")
    print(f"Initial state (after opening):")
    print(state)
    print(f"Move history from opening: {state.move_history}")
    
    # Get legal moves
    moves = rules.get_legal_moves(state)
    print(f"\nLegal moves after opening: {len(moves)}")
    
    # Play a few random moves
    rng = random.Random(42)
    for i in range(5):
        if state.is_terminal():
            print(f"\nGame ended!")
            break
        
        moves = rules.get_legal_moves(state)
        if not moves:
            break
            
        move = rng.choice(moves)
        state = rules.apply_move(state, move)
        print(f"\nMove {len(state.move_history)}: Player {state.current_player.opponent()} plays {move}")
    
    print(f"\nFinal state:")
    print(state)
    
    return True


def test_symmetric_opening():
    """Test symmetric opening rules."""
    print("\n" + "=" * 50)
    print("Testing Symmetric Opening Rules")
    print("=" * 50)
    
    rules = SymmetricOpeningRules(seed=42)
    state = rules.create_initial_state()
    
    print(f"Rule variant: {rules.name}")
    print(f"Initial state (after opening):")
    print(state)
    print(f"Move history from opening: {state.move_history}")
    
    return True


def test_game_completion():
    """Test that a game can be completed."""
    print("\n" + "=" * 50)
    print("Testing Game Completion (Random Playout)")
    print("=" * 50)
    
    rules = StandardRules()
    rng = random.Random(42)
    
    # Play multiple games
    results = {Player.X: 0, Player.O: 0, None: 0}
    game_lengths = []
    
    for game_num in range(10):
        state = rules.create_initial_state()
        move_count = 0
        
        while not state.is_terminal():
            moves = rules.get_legal_moves(state)
            if not moves:
                break
            move = rng.choice(moves)
            state = rules.apply_move(state, move)
            move_count += 1
        
        winner = state.get_winner()
        results[winner] += 1
        game_lengths.append(move_count)
        
        status = "Draw" if winner is None else f"{winner} wins"
        print(f"Game {game_num + 1}: {status} after {move_count} moves")
    
    print(f"\nResults: X wins: {results[Player.X]}, O wins: {results[Player.O]}, Draws: {results[None]}")
    print(f"Average game length: {sum(game_lengths) / len(game_lengths):.1f} moves")
    
    return True


def compare_variants():
    """Compare different rule variants."""
    print("\n" + "=" * 50)
    print("Comparing Rule Variants")
    print("=" * 50)
    
    variants = [
        StandardRules(),
        RandomizedOpeningRules(seed=42),
        SymmetricOpeningRules(seed=42),
    ]
    
    rng = random.Random(42)
    games_per_variant = 50
    
    for rules in variants:
        results = {Player.X: 0, Player.O: 0, None: 0}
        game_lengths = []
        move_freedoms = []  # Track average legal moves per turn
        
        for _ in range(games_per_variant):
            state = rules.create_initial_state()
            game_move_counts = []
            
            while not state.is_terminal():
                moves = rules.get_legal_moves(state)
                if not moves:
                    break
                game_move_counts.append(len(moves))
                move = rng.choice(moves)
                state = rules.apply_move(state, move)
            
            winner = state.get_winner()
            results[winner] += 1
            game_lengths.append(len(state.move_history))
            if game_move_counts:
                move_freedoms.append(sum(game_move_counts) / len(game_move_counts))
        
        x_rate = results[Player.X] / games_per_variant * 100
        o_rate = results[Player.O] / games_per_variant * 100
        draw_rate = results[None] / games_per_variant * 100
        avg_length = sum(game_lengths) / len(game_lengths)
        avg_freedom = sum(move_freedoms) / len(move_freedoms) if move_freedoms else 0
        
        print(f"\n{rules.name}:")
        print(f"  X win rate: {x_rate:.1f}%")
        print(f"  O win rate: {o_rate:.1f}%")
        print(f"  Draw rate: {draw_rate:.1f}%")
        print(f"  Avg game length: {avg_length:.1f} moves")
        print(f"  Avg move freedom: {avg_freedom:.1f} choices/turn")


if __name__ == "__main__":
    test_standard_rules()
    test_randomized_opening()
    test_symmetric_opening()
    test_game_completion()
    compare_variants()
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
