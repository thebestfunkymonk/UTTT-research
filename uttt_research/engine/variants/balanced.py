"""
Balanced Rule Variants for Ultimate Tic-Tac-Toe

These variants aim to eliminate first-player advantage through various mechanisms:
- Pie Rule: Allows P2 to swap sides after P1's first move
- Open Board: When sent to a WON board, can play in ANY ONGOING board OR ANY WON board
- Won-Board Play: Can play in WON boards (if not FULL) - treated like ONGOING when targeted
- Kill-Move: Constraints after winning a macro-board

Terminology:
- ONGOING: Board still not won and not full
- WON: Board won but still has empty cells (not full)
- FULL: Board has no empty cells (DRAW status)
"""
from typing import Optional, Tuple, List
from ..rules import UTTTRules
from ..state import UTTTState, Player, MacroBoardStatus


class PieRuleUTTT(UTTTRules):
    """
    UTTT with Pie Rule.
    
    After X's first move, O can choose to:
    1. Play normally (make their own move)
    2. Swap - take X's position and let X make a new first move
    
    This is stateless - pie phase is determined from game state.
    Swap move is encoded as (-1, -1) in move history.
    """
    
    @property
    def name(self) -> str:
        return "Pie Rule UTTT"
    
    def _is_pie_phase(self, state: UTTTState) -> bool:
        """Check if we're in pie phase (O can swap after X's first move)."""
        # Pie phase: exactly 1 move made, it's O's turn, and no swap has occurred
        if len(state.move_history) != 1:
            return False
        if state.current_player != Player.O:
            return False
        # Check that a swap hasn't already happened (swap is marked as (-1, -1))
        return state.move_history[0] != (-1, -1)
    
    def get_legal_moves(self, state: UTTTState) -> List[Tuple[int, int]]:
        """Get legal moves, including swap option during pie phase."""
        if state.is_terminal():
            return []
        
        # Normal moves
        moves = super().get_legal_moves(state)
        
        # During pie phase, O can swap
        if self._is_pie_phase(state):
            moves.append((-1, -1))
        
        return moves
    
    def apply_move(self, state: UTTTState, move: Tuple[int, int]) -> UTTTState:
        """Apply move, handling swap specially."""
        # Check for swap move
        if move == (-1, -1):
            if not self._is_pie_phase(state):
                raise ValueError("Swap move only valid during pie phase")
            
            # Swap: O takes X's position, X must make a new first move
            new_state = state.copy()
            
            # Swap the piece that was played
            macro_idx, cell_idx = state.move_history[0]
            new_state.set_cell(macro_idx, cell_idx, Player.O)
            
            # Mark swap in history
            new_state.move_history.append((-1, -1))
            
            # X plays next (making a "new" first move)
            new_state.current_player = Player.X
            new_state.target_macro = None  # X can play anywhere
            
            return new_state
        
        # Normal move
        return super().apply_move(state, move)


class OpenBoardRules(UTTTRules):
    """
    UTTT with more permissive targeting.
    
    When sent to a WON macro-board, the player can play in ANY ONGOING
    macro-board OR in ANY WON macro-board (including the one they were sent to).
    
    This variant allows for more strategic options near the end game.
    """
    
    @property
    def name(self) -> str:
        return "Open Board UTTT"
    
    def get_legal_moves(self, state: UTTTState) -> List[Tuple[int, int]]:
        """Get legal moves with open board rules."""
        if state.is_terminal():
            return []
        
        moves = []
        
        if state.target_macro is not None:
            target_status = state.get_macro_status(state.target_macro)
            
            if target_status == MacroBoardStatus.ONGOING:
                # Must play in target macro-board
                macro_indices = [state.target_macro]
            else:
                # Target is decided - can play anywhere (standard behavior)
                macro_indices = list(range(9))
        else:
            # Can play anywhere
            macro_indices = list(range(9))
        
        # Get empty cells in playable macro-boards
        for macro_idx in macro_indices:
            status = state.get_macro_status(macro_idx)
            # Only skip if FULL (DRAW status), allow ONGOING or WON boards
            if status == MacroBoardStatus.DRAW:
                continue
            for cell_idx in range(9):
                if state.get_cell(macro_idx, cell_idx) == Player.NONE:
                    moves.append((macro_idx, cell_idx))
        
        return moves


class WonBoardPlayRules(UTTTRules):
    """
    UTTT where moves can be made in WON (but not FULL) macro-boards.
    
    This changes the dynamics significantly:
    - WON boards can still have strategic value
    - When sent to a WON board, must play in that WON board (if not FULL)
    - Games tend to last longer
    - More cells are utilized on average
    """
    
    @property
    def name(self) -> str:
        return "Won-Board Play UTTT"
    
    def _is_macro_full(self, state: UTTTState, macro_idx: int) -> bool:
        """Check if a macro-board has no empty cells."""
        for cell_idx in range(9):
            if state.get_cell(macro_idx, cell_idx) == Player.NONE:
                return False
        return True
    
    def _is_board_full(self, state: UTTTState) -> bool:
        """Check if all 81 cells are filled."""
        for macro_idx in range(9):
            if not self._is_macro_full(state, macro_idx):
                return False
        return True
    
    def get_legal_moves(self, state: UTTTState) -> List[Tuple[int, int]]:
        """Get legal moves allowing play in WON boards."""
        # Check for game winner first (macro-board win)
        if state.check_game_winner() != MacroBoardStatus.ONGOING:
            return []
        
        # In this variant, game continues until board is FULL or won
        if self._is_board_full(state):
            return []
        
        moves = []
        
        # Determine playable macro-boards
        if state.target_macro is not None:
            if not self._is_macro_full(state, state.target_macro):
                # Target is not FULL - must play there (even if WON)
                macro_indices = [state.target_macro]
            else:
                # Target is FULL - can play in any non-FULL board
                macro_indices = [i for i in range(9) 
                               if not self._is_macro_full(state, i)]
        else:
            # Can play in any non-FULL macro-board (ONGOING or WON)
            macro_indices = [i for i in range(9) 
                           if not self._is_macro_full(state, i)]
        
        # Get empty cells
        for macro_idx in macro_indices:
            for cell_idx in range(9):
                if state.get_cell(macro_idx, cell_idx) == Player.NONE:
                    moves.append((macro_idx, cell_idx))
        
        return moves
    
    def get_next_target_macro(self, state: UTTTState, cell_idx: int) -> Optional[int]:
        """Next target allows WON boards if not FULL."""
        # Target is decided by cell_idx - check if it has empty cells (not FULL)
        # Can target ONGOING or WON boards, but not FULL boards
        if not self._is_macro_full(state, cell_idx):
            return cell_idx
        return None


class KillMoveRules(UTTTRules):
    """
    UTTT with Kill-Move constraint.
    
    If a move wins a macro-board, the next player must play in that
    same macro-board (if possible). This creates more tactical depth
    around winning boards.
    
    This is stateless - all targeting logic is computed from state.
    """
    
    @property
    def name(self) -> str:
        return "Kill-Move UTTT"
    
    def apply_move(self, state: UTTTState, move: Tuple[int, int]) -> UTTTState:
        """Apply move and check for kill-move constraint."""
        macro_idx, cell_idx = move
        
        # Get macro status before move
        old_status = state.get_macro_status(macro_idx)
        
        # Apply the move normally
        new_state = state.copy()
        new_state.set_cell(macro_idx, cell_idx, state.current_player)
        new_state.move_history.append(move)
        
        # Check if macro-board was just won
        new_macro_status = new_state.check_macro_winner(macro_idx)
        if new_macro_status != MacroBoardStatus.ONGOING:
            new_state.set_macro_status(macro_idx, new_macro_status)
        
        # Determine next target
        just_won = (old_status == MacroBoardStatus.ONGOING and 
                   new_macro_status in [MacroBoardStatus.X_WON, MacroBoardStatus.O_WON])
        
        if just_won:
            # Kill-move: opponent plays in the board matching the winning move's cell
            if new_state.get_macro_status(cell_idx) == MacroBoardStatus.ONGOING:
                new_state.target_macro = cell_idx
            else:
                new_state.target_macro = None
        else:
            # Standard targeting
            if new_state.get_macro_status(cell_idx) == MacroBoardStatus.ONGOING:
                new_state.target_macro = cell_idx
            else:
                new_state.target_macro = None
        
        new_state.current_player = state.current_player.opponent()
        return new_state


class BalancedRules(UTTTRules):
    """
    Combined balanced variant with multiple fairness mechanisms.
    
    This combines:
    1. Pie Rule option on move 1
    2. Open Board targeting: when sent to WON boards, can play in ANY ONGOING or WON board
    
    This is stateless and designed for MCTS compatibility.
    """
    
    @property
    def name(self) -> str:
        return "Balanced UTTT"
    
    def _is_pie_phase(self, state: UTTTState) -> bool:
        """Check if we're in pie phase (O can swap after X's first move)."""
        if len(state.move_history) != 1:
            return False
        if state.current_player != Player.O:
            return False
        return state.move_history[0] != (-1, -1)
    
    def get_legal_moves(self, state: UTTTState) -> List[Tuple[int, int]]:
        """Get legal moves with pie rule and open board."""
        if state.is_terminal():
            return []
        
        moves = []
        
        # Determine playable macro-boards (open board style)
        if state.target_macro is not None:
            target_status = state.get_macro_status(state.target_macro)
            if target_status == MacroBoardStatus.ONGOING:
                macro_indices = [state.target_macro]
            elif target_status in [MacroBoardStatus.X_WON, MacroBoardStatus.O_WON]:
                # Target is WON - can play in any ONGOING or WON macro-board
                macro_indices = list(range(9))
            else:
                # Can play in any ONGOING macro-board
                macro_indices = [i for i in range(9) 
                               if state.get_macro_status(i) == MacroBoardStatus.ONGOING]
        else:
            macro_indices = [i for i in range(9) 
                           if state.get_macro_status(i) == MacroBoardStatus.ONGOING]
        
        # Get empty cells
        for macro_idx in macro_indices:
            # Skip only FULL (DRAW) macro-boards
            if state.get_macro_status(macro_idx) == MacroBoardStatus.DRAW:
                continue
            for cell_idx in range(9):
                if state.get_cell(macro_idx, cell_idx) == Player.NONE:
                    moves.append((macro_idx, cell_idx))
        
        # Pie rule: after X's first move, O can swap
        if self._is_pie_phase(state):
            moves.append((-1, -1))  # Swap move
        
        return moves
    
    def apply_move(self, state: UTTTState, move: Tuple[int, int]) -> UTTTState:
        """Apply move with pie rule handling."""
        # Handle swap
        if move == (-1, -1):
            if not self._is_pie_phase(state):
                raise ValueError("Swap move only valid during pie phase")
            
            new_state = state.copy()
            macro_idx, cell_idx = state.move_history[0]
            new_state.set_cell(macro_idx, cell_idx, Player.O)
            new_state.move_history.append((-1, -1))
            new_state.current_player = Player.X
            new_state.target_macro = None
            return new_state
        
        return super().apply_move(state, move)
