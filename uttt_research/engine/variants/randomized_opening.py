"""
Randomized Opening Variant

Based on "A Practical Method for Preventing Forced Wins in Ultimate Tic-Tac-Toe"
(arXiv:2207.06239)

The key idea is to randomize the first few moves to disrupt predetermined
strategies and balance the first-player advantage.

Method: Generate a 5-digit number where each digit (0-8) determines the
placement of the first 4 moves. The 5th digit determines who goes first
after the opening phase (odd = X, even = O).
"""
import random
from typing import Optional, Tuple, List
from ..rules import UTTTRules
from ..state import UTTTState, Player, MacroBoardStatus


class RandomizedOpeningRules(UTTTRules):
    """
    UTTT with randomized opening moves to prevent forced wins.
    
    The opening consists of 4 pre-placed moves determined by random digits.
    This disrupts X's winning strategy and creates more balanced games.
    """
    
    def __init__(self, seed: Optional[int] = None, opening_code: Optional[str] = None):
        """
        Initialize with optional seed or specific opening code.
        
        Args:
            seed: Random seed for reproducibility
            opening_code: 5-digit string (0-8 each) to use as opening
                         If None, a random code is generated
        """
        self.seed = seed
        self._rng = random.Random(seed)
        self._opening_code = opening_code
        
    @property
    def name(self) -> str:
        return "Randomized Opening UTTT"
    
    def _generate_opening_code(self) -> str:
        """Generate a random 5-digit opening code."""
        return "".join(str(self._rng.randint(0, 8)) for _ in range(5))
    
    def _apply_opening(self, state: UTTTState, code: str) -> UTTTState:
        """
        Apply the opening moves based on the code.
        
        Code interpretation:
        - Digit 1: X's first move macro-board (cell is center=4)
        - Digit 2: O's first move (in the macro-board X sent them to, cell=digit)
        - Digit 3: X's second move (in the macro-board O sent them to, cell=digit)  
        - Digit 4: O's second move (in the macro-board X sent them to, cell=digit)
        - Digit 5: Determines starting player after opening (odd=X, even=O)
        
        Note: This is one interpretation. The actual paper may differ slightly.
        We'll use a simpler scheme that's easier to analyze:
        - Each digit pair (macro, cell) for 2 moves, then swap first player
        """
        # Simpler interpretation: 
        # Digits 0-1: Move 1 (macro=d0, cell=d1) by X
        # Digits 2-3: Move 2 (macro=d2, cell=d3) by O  
        # Digit 4: Starting player after setup (odd=X, even=O)
        
        # Actually, let's use an even simpler but effective scheme:
        # Pre-place 2 moves each in a symmetric pattern
        
        d = [int(c) for c in code]
        
        # Move 1: X plays in macro d[0], cell 4 (center)
        macro1 = d[0]
        cell1 = 4  # Always center for first move
        if state.get_cell(macro1, cell1) == Player.NONE:
            state.set_cell(macro1, cell1, Player.X)
            state.move_history.append((macro1, cell1))
            status = state.check_macro_winner(macro1)
            if status != MacroBoardStatus.ONGOING:
                state.set_macro_status(macro1, status)
        
        # Move 2: O plays in macro cell1 (=4), cell d[1]
        macro2 = cell1  # Sent to center macro
        cell2 = d[1]
        if state.get_cell(macro2, cell2) == Player.NONE:
            state.set_cell(macro2, cell2, Player.O)
            state.move_history.append((macro2, cell2))
            status = state.check_macro_winner(macro2)
            if status != MacroBoardStatus.ONGOING:
                state.set_macro_status(macro2, status)
        
        # Move 3: X plays in macro cell2, cell d[2]
        macro3 = cell2
        cell3 = d[2]
        if state.get_cell(macro3, cell3) == Player.NONE:
            state.set_cell(macro3, cell3, Player.X)
            state.move_history.append((macro3, cell3))
            status = state.check_macro_winner(macro3)
            if status != MacroBoardStatus.ONGOING:
                state.set_macro_status(macro3, status)
        
        # Move 4: O plays in macro cell3, cell d[3]
        macro4 = cell3
        cell4 = d[3]
        if state.get_cell(macro4, cell4) == Player.NONE:
            state.set_cell(macro4, cell4, Player.O)
            state.move_history.append((macro4, cell4))
            status = state.check_macro_winner(macro4)
            if status != MacroBoardStatus.ONGOING:
                state.set_macro_status(macro4, status)
        
        # Set target macro based on last move
        if state.get_macro_status(cell4) == MacroBoardStatus.ONGOING:
            state.target_macro = cell4
        else:
            state.target_macro = None
        
        # Digit 5 determines who plays next (odd = X, even = O)
        # This provides additional balance
        if d[4] % 2 == 1:
            state.current_player = Player.X
        else:
            state.current_player = Player.O
        
        return state
    
    def create_initial_state(self) -> UTTTState:
        """Create initial state with randomized opening."""
        state = UTTTState()
        
        # Use provided code or generate new one
        code = self._opening_code if self._opening_code else self._generate_opening_code()
        
        # Store the code for reference
        self._last_opening_code = code
        
        # Apply the opening
        return self._apply_opening(state, code)
    
    def get_opening_code(self) -> str:
        """Get the last used opening code."""
        return getattr(self, '_last_opening_code', None)


class SymmetricOpeningRules(UTTTRules):
    """
    Alternative balanced variant using symmetric opening.
    
    Both players get one move each in a symmetric position before
    normal play begins. This is simpler than the full randomized scheme.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self._rng = random.Random(seed)
    
    @property  
    def name(self) -> str:
        return "Symmetric Opening UTTT"
    
    def create_initial_state(self) -> UTTTState:
        """Create initial state with symmetric opening."""
        state = UTTTState()
        
        # X plays center of a random macro-board
        x_macro = self._rng.randint(0, 8)
        state.set_cell(x_macro, 4, Player.X)  # Center
        state.move_history.append((x_macro, 4))
        
        # O plays center of the opposite macro-board (point symmetry)
        o_macro = 8 - x_macro
        state.set_cell(o_macro, 4, Player.O)  # Center
        state.move_history.append((o_macro, 4))
        
        # X plays next, target is the center macro (where O sent them)
        state.current_player = Player.X
        state.target_macro = 4  # O played in center, sent X to center
        
        # Actually need to reconsider - the cell O played determines where X goes
        # O played cell 4, so X must play in macro 4
        if state.get_macro_status(4) == MacroBoardStatus.ONGOING:
            state.target_macro = 4
        else:
            state.target_macro = None
        
        return state
