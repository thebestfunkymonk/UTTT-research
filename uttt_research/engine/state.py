"""
UTTT Game State Representation

The board is a 9x9 grid organized as 9 macro-boards (3x3 each).
Each macro-board can be won by either player or drawn.
The game is won by winning 3 macro-boards in a row.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Tuple, List
import numpy as np


class Player(IntEnum):
    """Player enumeration. NONE represents empty cell."""
    NONE = 0
    X = 1
    O = 2
    
    def opponent(self) -> Player:
        if self == Player.X:
            return Player.O
        elif self == Player.O:
            return Player.X
        return Player.NONE
    
    def __str__(self) -> str:
        if self == Player.X:
            return "X"
        elif self == Player.O:
            return "O"
        return "."


class MacroBoardStatus(IntEnum):
    """Status of a macro-board."""
    ONGOING = 0
    X_WON = 1
    O_WON = 2
    DRAW = 3


# Win patterns for a 3x3 board (indices)
WIN_PATTERNS = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6)              # diagonals
]


@dataclass
class UTTTState:
    """
    Immutable game state for Ultimate Tic-Tac-Toe.
    
    Attributes:
        board: 9x9 numpy array of Player values
        macro_board: 3x3 numpy array of MacroBoardStatus values
        current_player: Player whose turn it is
        target_macro: Optional[int] - which macro-board must be played in (0-8), or None if any
        move_history: List of (macro_idx, cell_idx) tuples
    """
    board: np.ndarray = field(default_factory=lambda: np.zeros((9, 9), dtype=np.int8))
    macro_board: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.int8))
    current_player: Player = Player.X
    target_macro: Optional[int] = None
    move_history: List[Tuple[int, int]] = field(default_factory=list)
    
    def copy(self) -> UTTTState:
        """Create a deep copy of the state."""
        return UTTTState(
            board=self.board.copy(),
            macro_board=self.macro_board.copy(),
            current_player=self.current_player,
            target_macro=self.target_macro,
            move_history=self.move_history.copy()
        )
    
    def get_cell(self, macro_idx: int, cell_idx: int) -> Player:
        """Get the value of a cell within a macro-board."""
        macro_row, macro_col = divmod(macro_idx, 3)
        cell_row, cell_col = divmod(cell_idx, 3)
        row = macro_row * 3 + cell_row
        col = macro_col * 3 + cell_col
        return Player(self.board[row, col])
    
    def set_cell(self, macro_idx: int, cell_idx: int, player: Player) -> None:
        """Set the value of a cell within a macro-board (mutates state)."""
        macro_row, macro_col = divmod(macro_idx, 3)
        cell_row, cell_col = divmod(cell_idx, 3)
        row = macro_row * 3 + cell_row
        col = macro_col * 3 + cell_col
        self.board[row, col] = player
    
    def get_macro_cells(self, macro_idx: int) -> np.ndarray:
        """Get all 9 cells of a macro-board as a flat array."""
        macro_row, macro_col = divmod(macro_idx, 3)
        start_row = macro_row * 3
        start_col = macro_col * 3
        return self.board[start_row:start_row+3, start_col:start_col+3].flatten()
    
    def get_macro_status(self, macro_idx: int) -> MacroBoardStatus:
        """Get the status of a macro-board."""
        macro_row, macro_col = divmod(macro_idx, 3)
        return MacroBoardStatus(self.macro_board[macro_row, macro_col])
    
    def set_macro_status(self, macro_idx: int, status: MacroBoardStatus) -> None:
        """Set the status of a macro-board (mutates state)."""
        macro_row, macro_col = divmod(macro_idx, 3)
        self.macro_board[macro_row, macro_col] = status
    
    def check_macro_winner(self, macro_idx: int) -> MacroBoardStatus:
        """Check if a macro-board has been won/drawn."""
        cells = self.get_macro_cells(macro_idx)
        
        # Check for wins
        for pattern in WIN_PATTERNS:
            values = [cells[i] for i in pattern]
            if values[0] != Player.NONE and values[0] == values[1] == values[2]:
                if values[0] == Player.X:
                    return MacroBoardStatus.X_WON
                else:
                    return MacroBoardStatus.O_WON
        
        # Check for draw (all cells filled)
        if all(c != Player.NONE for c in cells):
            return MacroBoardStatus.DRAW
        
        return MacroBoardStatus.ONGOING
    
    def check_game_winner(self) -> MacroBoardStatus:
        """Check if the game has been won/drawn."""
        macro_flat = self.macro_board.flatten()
        
        # Check for wins
        for pattern in WIN_PATTERNS:
            values = [macro_flat[i] for i in pattern]
            if all(v == MacroBoardStatus.X_WON for v in values):
                return MacroBoardStatus.X_WON
            if all(v == MacroBoardStatus.O_WON for v in values):
                return MacroBoardStatus.O_WON
        
        # Check for draw (all macro-boards decided)
        if all(m != MacroBoardStatus.ONGOING for m in macro_flat):
            return MacroBoardStatus.DRAW
        
        return MacroBoardStatus.ONGOING
    
    def is_terminal(self) -> bool:
        """Check if the game is over."""
        return self.check_game_winner() != MacroBoardStatus.ONGOING
    
    def get_winner(self) -> Optional[Player]:
        """Get the winner of the game, or None if ongoing/draw."""
        status = self.check_game_winner()
        if status == MacroBoardStatus.X_WON:
            return Player.X
        elif status == MacroBoardStatus.O_WON:
            return Player.O
        return None
    
    def __str__(self) -> str:
        """Pretty print the board."""
        lines = []
        for macro_row in range(3):
            for cell_row in range(3):
                row_str = ""
                for macro_col in range(3):
                    macro_idx = macro_row * 3 + macro_col
                    for cell_col in range(3):
                        cell_idx = cell_row * 3 + cell_col
                        cell = self.get_cell(macro_idx, cell_idx)
                        row_str += str(cell)
                    if macro_col < 2:
                        row_str += "|"
                lines.append(row_str)
            if macro_row < 2:
                lines.append("-" * 11)
        
        # Add macro-board status
        lines.append("")
        lines.append(f"Macro board status:")
        for row in range(3):
            row_str = ""
            for col in range(3):
                status = MacroBoardStatus(self.macro_board[row, col])
                if status == MacroBoardStatus.X_WON:
                    row_str += "X"
                elif status == MacroBoardStatus.O_WON:
                    row_str += "O"
                elif status == MacroBoardStatus.DRAW:
                    row_str += "D"
                else:
                    row_str += "."
            lines.append(row_str)
        
        lines.append(f"\nCurrent player: {self.current_player}")
        lines.append(f"Target macro: {self.target_macro}")
        
        return "\n".join(lines)
