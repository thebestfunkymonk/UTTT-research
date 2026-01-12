# Understanding the Differences: Standard vs Open Board vs Won-Board Play

## Terminology

**Important:** Throughout this document, we use these terms consistently:
- **ONGOING**: A macro-board that is still not won and not full (has empty cells, no winner yet)
- **WON**: A macro-board that has been won by a player but still has empty cells (not full)
- **FULL**: A macro-board with no remaining empty cells (DRAW status - all 9 cells filled)

---

## The Core Question: "Where Can I Play?"

All three variants use the same targeting rule: **the cell you play determines which macro-board your opponent must play in next**. The difference is what happens when you're sent to a **WON** macro-board.

---

## Scenario Setup

Imagine this situation:
- Macro-board 4 (center) is **WON by X** (has 3 empty cells - not FULL)
- Macro-board 1 is **ONGOING** (still not won, not full)
- Macro-board 7 is **ONGOING** (still not won, not full)
- You just played in cell 4, so your opponent is **sent to macro-board 4**

---

## Standard UTTT

**Rule:** Can only play in **ONGOING** macro-boards.

**What happens:**
- Opponent is sent to macro-board 4 (which is WON)
- Since macro-board 4 is WON (not ONGOING), opponent **cannot play there**
- Opponent gets **free choice** - can play in ANY ONGOING macro-board (1, 7, or any other ONGOING board)

**Key Point:** WON boards are "off limits" - you can't play inside them, but being sent to one gives you freedom to play elsewhere.

---

## Open Board UTTT

**Rule:** When sent to a **WON** board, you can play in ANY ONGOING board OR in ANY WON board (including the one you were sent to).

**What happens:**
- Opponent is sent to macro-board 4 (which is WON)
- Opponent can play in macro-board 4 (WON but not FULL) ✅
- Opponent can play in ANY ONGOING macro-board (1, 7, or any other ONGOING board) ✅
- Opponent can play in ANY OTHER WON board (if it has empty cells) ✅

**Key difference:** You CAN play inside WON boards if they have empty cells, AND you can play in ANY WON board, not just the one you were sent to.

---

## The Actual Differences (Based on Code)

### Standard UTTT
- **Can play in:** Only ONGOING macro-boards
- **When sent to WON board:** Can play in ANY ONGOING macro-board
- **Cannot play:** Inside WON boards (even if they have empty cells)

### Open Board UTTT  
- **Can play in:** ONGOING macro-boards OR WON boards (if not FULL)
- **When sent to WON board:** Can play in ANY ONGOING board OR in ANY WON board (including the one you were sent to)
- **Key difference:** You CAN play inside WON boards if they have empty cells

### Won-Board Play UTTT
- **Can play in:** ANY macro-board that's not FULL (ONGOING OR WON)
- **When sent to WON board:** Must play in that WON board (if it has empty cells)
- **Key difference:** WON boards are treated like ONGOING boards - you can and must play in them when targeted

---

## Concrete Example

**Situation:**
- You play in cell 4, sending opponent to macro-board 4
- Macro-board 4 is **WON by X** but has 2 empty cells (WON, not FULL)
- Macro-board 1 is **ONGOING** with 5 empty cells
- Macro-board 7 is **ONGOING** with 3 empty cells
- Macro-board 2 is **WON by O** but has 1 empty cell (WON, not FULL)

**What can opponent do?**

### Standard UTTT:
- ❌ Cannot play in macro-board 4 (it's WON)
- ❌ Cannot play in macro-board 2 (it's WON)
- ✅ Can play in macro-board 1 (ONGOING)
- ✅ Can play in macro-board 7 (ONGOING)
- ✅ Can play in any other ONGOING macro-board

**Result:** Opponent has freedom to choose any ONGOING board.

### Open Board UTTT:
- ✅ **CAN** play in macro-board 4 (WON but not FULL) ← **KEY DIFFERENCE**
- ✅ **CAN** play in macro-board 2 (WON but not FULL) ← **Can play in ANY WON board**
- ✅ Can play in macro-board 1 (ONGOING)
- ✅ Can play in macro-board 7 (ONGOING)
- ✅ Can play in any other ONGOING macro-board

**Result:** Opponent has MORE options - can play in ANY WON board OR any ONGOING board.

### Won-Board Play UTTT:
- ✅ **MUST** play in macro-board 4 (WON but not FULL) ← **KEY DIFFERENCE**
- ❌ Cannot play in macro-board 1 (not the target)
- ❌ Cannot play in macro-board 7 (not the target)
- ❌ Cannot play in macro-board 2 (not the target)

**Result:** Opponent is still constrained to the target board, even though it's WON.

---

## Visual Summary

```
Situation: Sent to macro-board 4 (WON by X, 2 empty cells - not FULL)
Also assume: Macro-board 2 is WON by O (1 empty cell - not FULL)

Standard UTTT:
  Macro 4 (WON):     ❌ Cannot play
  Macro 2 (WON):     ❌ Cannot play
  Macro 1 (ONGOING): ✅ Can play
  Macro 7 (ONGOING): ✅ Can play
  → Freedom: Play anywhere ONGOING

Open Board UTTT:
  Macro 4 (WON):     ✅ CAN play (the one you were sent to)
  Macro 2 (WON):     ✅ CAN play (ANY WON board!)
  Macro 1 (ONGOING): ✅ Can play
  Macro 7 (ONGOING): ✅ Can play
  → Freedom: Play in ANY WON board OR anywhere ONGOING

Won-Board Play UTTT:
  Macro 4 (WON):     ✅ MUST play (still constrained to target!)
  Macro 2 (WON):     ❌ Cannot play (not the target)
  Macro 1 (ONGOING): ❌ Cannot play (not the target)
  Macro 7 (ONGOING): ❌ Cannot play (not the target)
  → Constraint: Must play in target (even if WON)
```

---

## Why These Differences Matter

### Standard UTTT
- **Philosophy:** WON boards are "dead" - no strategic value
- **Effect:** Once a board is WON, it's ignored (even if not FULL)
- **Gameplay:** More tactical, focused on winning boards

### Open Board UTTT
- **Philosophy:** WON boards still have empty cells, so they matter
- **Effect:** More strategic options, less "railroading"
- **Gameplay:** More freedom, can fill WON boards for positioning

### Won-Board Play UTTT
- **Philosophy:** WON boards are just "not FULL" - treat them like ONGOING boards when targeted
- **Effect:** Games last longer, more cells utilized
- **Gameplay:** More complex, WON boards can be "contested" even after being won

---

## Performance Impact

From the analysis:

| Metric | Standard | Open Board | Won-Board Play |
|--------|----------|------------|----------------|
| **Move Freedom** | 9.4 choices/turn | 10.7 choices/turn | 6.7 choices/turn |
| **Game Length** | 58 moves | 64 moves | 68 moves |
| **Fairness** | 91.8% | 93.3% | 80.5% |

**Open Board** gives the most freedom (10.7 choices/turn) because you can play in WON boards OR ONGOING boards.

**Won-Board Play** has less freedom (6.7 choices/turn) because you're still constrained to the target board, but games are longer (68 moves) because you keep playing in WON boards.

---

## The Confusion

The names are a bit misleading:
- **"Open Board"** sounds like it should be the most permissive, but it's actually in the middle
- **"Won-Board Play"** sounds like you can play in won boards freely, but you're still constrained by targeting rules

The key insight: **Open Board** gives you the OPTION to play in ANY WON board (including the one you were sent to), while **Won-Board Play** still constrains you to the target board (even if it's WON).
