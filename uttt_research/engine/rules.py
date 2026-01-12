"""
Abstract base class for UTTT game rules.

Different variants can override specific methods to change game behavior.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from .state import UTTTState, Player, MacroBoardStatus


class UTTTRules(ABC):
    """
    Abstract base class defining the interface for UTTT rule variants.
    
    Subclasses can override specific methods to implement rule variations
    while keeping the core game logic intact.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this rule variant."""
        pass
    
    def create_initial_state(self) -> UTTTState:
        """
        Create the initial game state.
        
        Override this to implement different starting positions
        (e.g., randomized openings).
        """
        return UTTTState()
    
    def get_legal_moves(self, state: UTTTState) -> List[Tuple[int, int]]:
        """
        Get all legal moves for the current player.
        
        Returns:
            List of (macro_idx, cell_idx) tuples representing legal moves.
        """
        if state.is_terminal():
            return []
        
        moves = []
        
        # Determine which macro-boards can be played in
        # Standard rules: can only play in ONGOING boards
        if state.target_macro is not None:
            # Must play in the target macro-board (if it's ONGOING)
            macro_indices = [state.target_macro]
        else:
            # Can play in any ONGOING macro-board
            macro_indices = [i for i in range(9) 
                           if state.get_macro_status(i) == MacroBoardStatus.ONGOING]
        
        # Get empty cells in the playable macro-boards
        for macro_idx in macro_indices:
            # Standard rules: only allow ONGOING boards (skip WON and FULL)
            if state.get_macro_status(macro_idx) != MacroBoardStatus.ONGOING:
                continue
            for cell_idx in range(9):
                if state.get_cell(macro_idx, cell_idx) == Player.NONE:
                    moves.append((macro_idx, cell_idx))
        
        return moves
    
    def apply_move(self, state: UTTTState, move: Tuple[int, int]) -> UTTTState:
        """
        Apply a move to the state and return the new state.
        
        Args:
            state: Current game state
            move: (macro_idx, cell_idx) tuple
            
        Returns:
            New game state after the move
        """
        macro_idx, cell_idx = move
        
        # Create new state
        new_state = state.copy()
        
        # Place the piece
        new_state.set_cell(macro_idx, cell_idx, state.current_player)
        new_state.move_history.append(move)
        
        # Check if this macro-board is now won/drawn
        macro_status = new_state.check_macro_winner(macro_idx)
        if macro_status != MacroBoardStatus.ONGOING:
            new_state.set_macro_status(macro_idx, macro_status)
        
        # Determine next target macro-board
        new_state.target_macro = self.get_next_target_macro(new_state, cell_idx)
        
        # Switch player
        new_state.current_player = state.current_player.opponent()
        
        return new_state
    
    def get_next_target_macro(self, state: UTTTState, cell_idx: int) -> Optional[int]:
        """
        Determine the next target macro-board based on the cell played.
        
        Override this to implement different targeting rules.
        
        Args:
            state: The state after the move was made (but before player switch)
            cell_idx: The cell index that was played (0-8)
            
        Returns:
            The macro-board index the next player must play in, or None if any.
        """
        # Standard rule: next player must play in macro-board matching cell_idx
        # Only target if the board is ONGOING (not WON or FULL)
        if state.get_macro_status(cell_idx) == MacroBoardStatus.ONGOING:
            return cell_idx
        else:
            # Target macro is WON or FULL, player can play anywhere ONGOING
            return None
    
    def is_move_legal(self, state: UTTTState, move: Tuple[int, int]) -> bool:
        """Check if a move is legal."""
        return move in self.get_legal_moves(state)
    
    def get_result(self, state: UTTTState, player: Player) -> float:
        """
        Get the game result from a player's perspective.
        
        Returns:
            1.0 for win, 0.0 for loss, 0.5 for draw, None if not terminal.
        """
        if not state.is_terminal():
            return None
        
        winner = state.get_winner()
        if winner is None:
            return 0.5  # Draw
        elif winner == player:
            return 1.0  # Win
        else:
            return 0.0  # Loss
