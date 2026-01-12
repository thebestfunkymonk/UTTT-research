"""
Standard Ultimate Tic-Tac-Toe Rules

This is the base implementation with no modifications.
X has a proven winning strategy in standard UTTT.
"""
from ..rules import UTTTRules


class StandardRules(UTTTRules):
    """
    Standard UTTT rules.
    
    - X moves first
    - The cell you play in determines the macro-board your opponent must play in
    - Can only play in ONGOING macro-boards
    - If sent to a WON or FULL macro-board, opponent can play in ANY ONGOING board
    - Win by getting 3 macro-boards in a row
    
    Terminology:
    - ONGOING: Board still not won and not full
    - WON: Board won but still has empty cells (not full)
    - FULL: Board has no empty cells (DRAW status)
    """
    
    @property
    def name(self) -> str:
        return "Standard UTTT"
