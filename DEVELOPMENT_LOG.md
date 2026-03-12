# Development Log

Running record of decisions, discoveries, and changes made phase by phase.

---

## Phase 1 — Hero Card OCR Pipeline

### Goal
Detect when new hole cards are dealt and immediately read + emit them as structured data in <100ms.

### Files Created
| File | Purpose |
|------|---------|
| `models.py` | `Card`, `HeroCardsResult` dataclasses |
| `config.py` | `TrackerConfig`, `RegionConfig`, `load_config()`, `crop_region()` |
| `capture.py` | macOS Quartz screenshot — pure functions |
| `card_detector.py` | Variance gate → NN split-crop → confidence → `HeroCardsResult` |
| `hero_watcher.py` | Polling loop, debounce (2 frames), dedup, `replay_dir()` |
| `test_hero_cards.py` | Accuracy test vs `label_hands.txt` ground truth |
| `NN/` | Copied `nn_card_reader.py` + `.pth` weights from old project |
| `regions.json` | Calibrated fractional region coordinates |

### Key Discovery: Two Hero Regions
`regions.json` hero region `[0.425, 0.6285, 0.5719, 0.6993]` is calibrated for the
**variance gate only**. The NN was trained on `[0.38, 0.55, 0.58, 0.75]` (larger area).
Using the calibrated region for the NN produced very low confidence (0.2–0.6).
**Fix**: Added `hero_nn` key to `regions.json` with the training fractions. Config uses
`hero` for the fast gate, `hero_nn` for the NN classifier.

### Key Discovery: Inset Cropping
The original `vision_tracker.py` doesn't split the hero region at midpoint — it uses
inset fractions: left card = `(cw//8, 5, cw//2-5, ch-5)`, right = `(cw//2+5, 5, 7*cw//8, ch-5)`.
Using exact midpoint split reduced confidence. Matched the original approach.

### Confidence Metric
`confidence = min(softmax_rank.max() * softmax_suit.max())` per card, then `min(card1, card2)`.
Default threshold: `0.30` (product of two softmax peaks — appears low but works well in practice).

### Results
- **602 / 618** images detected (97.4%)
- **157 / 157** labeled images correct (100%)
- **37ms/frame** average in replay mode

---

## Phase 2 — Table State Detection

### Planning Decisions

| Question | Options considered | Decision | Reason |
|---|---|---|---|
| Dealer button detection | Template match / color / OCR | 6 labeled seat regions + template match | Most accurate; region-based is faster than full-image sliding window |
| Hero seat identification | Iterate seats / manual config / fixed region | OCR `hero_name` region | Hero always at same position; one read, no matching needed |
| OCR tool | Tesseract / EasyOCR / hybrid | Tesseract | Already installed in old project; good for numbers/short names |
| Seat status detection | Template + OCR / OCR only | OCR only | "Empty Seat" and "Sitting Out" are text — room-agnostic approach |
| Scope | Snapshot vs continuous | Snapshot at HAND_START | Simpler; continuous tracking deferred to Phase 4 |
| Test strategy | Ground-truth labels / sanity / code-first | Code first → verify → label | See real output before committing to labeling |

### Files Created
| File | Purpose |
|------|---------|
| `table_state.py` | `PlayerState`, `TableState` dataclasses + `assign_positions()` + `clockwise_seat_order()` |
| `ocr.py` | `read_text()`, `read_number()`, `preprocess()` — Tesseract wrappers |
| `dealer_detector.py` | `DealerDetector` — NCC template match across all dealer regions |
| `table_scanner.py` | `TableScanner.scan()` — orchestrates all detections → `TableState` |
| `test_table_state.py` | Print / label / test modes |
| `mark_dealer_regions.py` | GUI to label dealer button regions per seat + hero |
| `README.md` | Setup, calibration, running tests |

### Files Modified
| File | Change |
|------|--------|
| `config.py` | Added `hero_name`, `hero_stack`, `hero_dealer`, `hero_bet`, `hero_seat` to `RegionConfig`; added `dealer_match_threshold`, `dealer_scales` to `TrackerConfig` |
| `regions.json` | Added `dealer` sub-region to each seat (via GUI tool); added top-level `hero_dealer`, `hero_bet` |
| `requirements.txt` | Added `pytesseract` |

### Key Design: Clockwise Seat Order
Position assignment (BTN/SB/BB/UTG/MP/CO) requires knowing the clockwise order of seats.
Rather than hardcoding, `clockwise_seat_order()` auto-computes from seat name region centers:
1. Find centroid of all seat positions (including hero)
2. For each seat: `angle = atan2(dx, -dy)` in screen coords (y-down, clockwise from top)
3. Sort by angle

Computed order for this table: `[4, 5, 0(hero), 1, 2, 3]` — verified correct.

### Position Assignment Formula
For N active players, chart = `_POSITION_CHARTS[N]` (e.g. `["UTG","MP","CO","BTN","SB","BB"]`).
BTN index in chart = `max(0, N-3)`. Assignment:
```
for chart_offset, pos in enumerate(chart):
    seat = active[(dealer_idx + chart_offset - btn_in_chart) % N]
    positions[seat] = pos
```
Verified correct for N = 2, 3, 4, 5, 6.

### Seat Status Rules
```
username ≈ "empty" AND stack ≈ "seat"  → EMPTY
stack contains "sitting" / "sit out"   → SITTING_OUT
username is blank                       → EMPTY
otherwise                               → ACTIVE
```
All OCR-based — no templates needed — works across different room skins.

### Known OCR Noise (to tune/fix in later phases)
- Action labels ("Fold", "Call", "Post SB") bleed into name region when a player acts
- Pipe characters (`|`) appear from UI border artifacts → strip in post-processing
- Some stacks show `?` (None) when stack region is partially obscured by chips
- Usernames with slashes: "Brownson76" → "Brownson/6" (tesseract confuses 7 and /)

### Results (Phase 2 — Print Mode)
- Dealer seat detected correctly on all 5 inspected images
- Position assignment correct (BB, UTG, MP, CO, BTN, SB all right)
- SITTING_OUT correctly detected from "Sitting Out" stack text
- Speed: ~800ms/frame (Tesseract dominates — acceptable for snapshot mode)

### Deferred to Phase 4
- `TableState.big_blind` and `small_blind` are `None` — will be detected from blind posts
- Continuous re-scan of stacks (see README Future Improvements)

---

---

## Phase 3 — Player Action Tracking

### Files Created
| File | Purpose |
|------|---------|
| `core/hand_state.py` | `HandState`, `PlayerAction`, `ActionRound`, `Street` dataclasses |
| `detection/action_detector.py` | Stack delta → `PlayerAction`, fold via cards variance, 2-frame debounce |
| `detection/street_detector.py` | Board slot variance → `Street` (PREFLOP/FLOP/TURN/RIVER), debounced |
| `tracking/game_tracker.py` | Unified poller: IDLE → HAND_STARTING → HAND_ACTIVE → HAND_COMPLETE |
| `test_game_tracker.py` | Full pipeline replay test with event log + summary table |

### Architecture Decisions
- **Stack delta as primary signal**: Name OCR dropped — saves 280ms/31% per frame; stacks + bets only = ~620ms/frame
- **Hero fold does not end hand**: `hero_folded=True` flag; tracking continues until board clears (valuable for opponent profiling)
- **YOUR_TURN via saturation gate**: RGB saturation spike in action button region (~1ms) — Fold/Call/Raise buttons have high colour saturation
- **Unified poller (not separate threads)**: Single `process_frame()` call handles all detection per tick; simpler to reason about, no race conditions
- **Tracking continues after fold**: All opponent actions post-fold captured in `all_actions`

### State Machine
```
IDLE → (hero cards debounced) → HAND_STARTING → (Phase 2 snapshot) → HAND_ACTIVE
HAND_ACTIVE → (board clears + hero absent ≥3 frames) → HAND_COMPLETE → IDLE
```

---

## Phase 4 — BB Detection & amount_bb

### Files Created
| File | Purpose |
|------|---------|
| `detection/bb_detector.py` | `parse_bb_from_title()` + `detect_bb_from_pot()` + `detect_bb()` |

### Files Modified
| File | Change |
|------|--------|
| `core/hand_state.py` | Added `bb_amount: Optional[float]` field |
| `tracking/game_tracker.py` | Detects BB at hand start, fills `amount_bb` on every `PlayerAction` |

### Detection Strategy
1. **Window title** (primary): Parse "SB/BB" pattern from PS window title — `kCGWindowName` already captured in `get_pokerstars_window()`. Supports `$0.50/$1.00`, `50/100`, `€1/€2` formats.
2. **Pot OCR fallback**: After blinds post, pot = SB + BB = 1.5 × BB, so `BB ≈ pot / 1.5`. Room-agnostic.

`amount_bb = action.amount / bb_amount` filled inline via `_fill_bb()` in `GameTracker`.

---

## Phase 5 — Community Card OCR

### Files Created
| File | Purpose |
|------|---------|
| `detection/board_detector.py` | `detect_board_cards(img, cfg)` — NN classify each board slot |

### Files Modified
| File | Change |
|------|--------|
| `tracking/game_tracker.py` | Calls `detect_board_cards()` on street change, fills `hand_state.community_cards` |
| `test_game_tracker.py` | Shows board cards + BB in HAND_COMPLETE line and summary table |

### Approach
- Same MobileNetV2 NN as hero cards — board cards are face-up playing cards with identical design
- Variance gate per slot → NN classify → confidence threshold 0.20 (lower than hero due to cleaner crops)
- Called on `on_street_change` so community_cards always reflects the current board state
- `board_cards_to_str()` utility for display

---

---

## Post-Phase 5 — Card Detection Accuracy & Tooling

### Problem: Board card detection at 20–30% accuracy

Board slot crops (~60×81 px, aspect ratio ~0.74) were classified by a model trained
exclusively on hero card crops (~67×126, ratio ~0.53). After resizing both to 128×192,
the distortion difference caused consistent rank misclassifications (8→6, 5→3, K→4, etc.).
Suit detection was unaffected because suit symbols are more distinctive shapes.

### Fix: Combined training dataset

Created `NN/train_card_model.py` — trains `card_detector_nn.pth` on both:
- **Hero crops**: extracted from `tests/` + `label_hands.txt` (618 images × 2 cards)
- **Board crops**: extracted from `captures/` + `label_cards.json` (labeled board slots)

Board crops are repeated to balance the dataset. Result after 60 epochs on MPS:
- Hero accuracy: **100%** (23/23 labeled images)
- Board accuracy: **100%** (23/23 labeled images, up from 30%)

### Files Created
| File | Purpose |
|------|---------|
| `label_cards.py` | Interactive labeller — opens image, shows detection, prompts hero + board |
| `test_cards.py` | Accuracy test with separate Hero? / Board? columns and per-image confidence |
| `NN/train_card_model.py` | Combined hero + board training script |

### Key Design: Label format
`label_cards.json` stores `{"filename": {"hero": ["Qs","5d"] | null, "board": [...] | null}}`.
Incremental saves after every image. `--redo` flag re-labels already-labelled images.

---

## Post-Phase 5 — Logger Refactor

All ANSI formatting and event handler functions extracted from `main.py` into
`tracking/logger.py` as a `PokerLogger` class.

| Before | After |
|--------|-------|
| Module-level colour constants + free functions in `main.py` | `PokerLogger` class with `verbose` attribute |
| `_verbose` global | `logger.verbose` instance attribute |
| `_TeeWriter` + `_setup_log_file()` in `main.py` | `logger.setup_log_file()` method |
| `on_hand_start` etc. as free functions | `logger.on_hand_start` etc. as methods |

`PokerLogger.colour(text, name)` is a public static method reused by `test_cards.py`
(replacing that file's own ANSI constant block).

Live event output additions:
- `HERO CARDS` line on hand start — bold, suit-coloured (red for h/d, white for s/c)
- Flop/Turn/River separator lines show board cards inline with suit colouring
- `--verbose` action lines now include `street total XBB ($Y)` alongside stack delta

---

## Post-Phase 5 — Street-Level Bet Tracking

### Problem
`action.amount` records chips added in a single action. For a raise → reraise → call
sequence on a street, there was no way to know a seat's total investment in the pot
for that street.

### Implementation

Added `street_total` / `street_total_bb` / `street_total_dollars` to `PlayerAction`.

`ActionDetector` tracks `_street_committed: Dict[int, float]` — total chips put in
per seat this street. Updated on every BET/CALL/RAISE/ALL_IN. Reset in
`reset_street_bets()` on street change and in `reset()` on hand end.

CHECK and FOLD carry the seat's current `street_total` (no new chips added, but the
running total is still useful context for the consumer).

`GameTracker._fill_bb()` fills `street_total_bb` and `street_total_dollars` using
the same `bb_amount` multiplier as `amount_dollars`.

---

## Post-Phase 5 — Detection Performance Optimisations

### Motivation
At 150ms poll rate, every frame was running all detectors regardless of whether the
screen had changed. Most frames during a hand are static (players thinking, animations
settling). The goal: eliminate redundant work without changing detection accuracy.

### 1 — Region hash cache (`detection/region_cache.py`)

`RegionCache` stores an 8×8 grayscale thumbnail hash per region coordinate tuple.
`cache.changed(img, region)` returns `True` only when pixels differ from the previous
call. First call always returns `True` (safe default).

**Rule**: stack and bet regions are never passed to the cache — they are the primary
action signal and must always be re-read.

Applied to:
- **Fold detection** (`action_detector.py`): `_has_cards()` variance check skipped
  when the cards region hash is unchanged. On static frames this is ~0ms instead of
  a numpy variance op per seat.
- **Board slot variance** (`street_detector.py`): each of the 5 slots tracks its own
  hash. Variance check skipped when slot pixels are identical to the last frame. Last
  known presence is reused. Eliminates 5 numpy ops per frame when board is static.
- **Action button saturation** (`game_tracker.py`): `_action_buttons_visible()` skipped
  when the button region is unchanged. Previous visibility state reused instead.

### 2 — Action detection runs on every frame

Action detection (`_actions.detect()`) runs unconditionally on every frame, including
preflop and while action buttons are visible. An earlier attempt to skip detection while
buttons were on screen caused missed preflop actions (blind posts, raises before hero acts)
and was reverted.

Community card detection is the only thing that is gated — it only runs on confirmed
street transitions (`changed=True`), never on every frame.

### 3 — Parallel stack OCR (reverted)

A `ThreadPoolExecutor` was added to read all seat stacks concurrently. It was removed
because the threading overhead introduced timing issues and made debugging harder with
no measurable benefit at the 150ms poll rate. The sequential loop is simpler and reliable.

### Files Created / Modified

| File | Change |
|------|--------|
| `detection/region_cache.py` | New — `RegionCache` class |
| `detection/action_detector.py` | Added `RegionCache` for fold detection only; sequential stack reads retained |
| `detection/street_detector.py` | Added `RegionCache` per board slot in `StreetDetector` |
| `tracking/game_tracker.py` | Added `RegionCache` for button region (YOUR_TURN detection only) |

---

---

## Post-Phase 5 — Threading, Fold Tracking & CHECK Detection

### Parallel stack OCR
`ActionDetector.detect()` now pre-crops all active seat stack regions and reads them concurrently via `ThreadPoolExecutor`. Only the Tesseract OCR calls are parallelised — action processing (`_check_delta`, `_street_committed`, `_current_call_amount`) remains sequential to avoid races on shared state. Folded seats are excluded from the job map before launching threads so out-of-order completion never corrupts seat→result mapping.

### Hero fold tracking
Previously `on_hand_complete` fired ~450ms after a preflop fold because `HERO_ABSENT_LIMIT=3` triggered before opponents finished. Fixed:
- `hero_folded=True` is set immediately when hero cards vanish (any street)
- After fold, hand ends only on natural signals: board clears after being seen (postflop), or new hero cards appear (preflop / everyone-folded-out scenario)
- No frame-count timeout when `hero_folded=True`

### CHECK detection
Removed `_infer_checks` (action-ordering inference). Replaced with direct blue-label detection:
- `_is_check_label(img, region)` counts blue-dominant pixels (`B > 80` and `B > R+30` and `B > G+20`). Fraction ≥ 0.02 → "Check" label present. No template file required.
- Runs per frame on each opponent `name` region via `RegionCache` (cost ~0 on static frames)
- Also runs on `hero_name` region for hero CHECK detection
- `check_label_active` flag per seat prevents duplicate emissions while label stays on screen
- Hero FOLD: emitted explicitly when buttons disappear and hero cards are gone (delta=0 → `_classify` returns None, so fold is detected via card-absence rather than stack delta)
- Hero BET/CALL/RAISE: falls back to `last_hero_stack()` when `_hero_stack_before` is None (buttons appeared and disappeared within one poll)

---

## Post-Phase 5 — Session State & JSONL Logging

### Files Created
| File | Purpose |
|------|---------|
| `tracking/session_state.py` | `SessionState` — username registry, JSONL log writer, `parse_table_name()` |
| `session_logs/` | Per-session JSONL files (created at runtime) |

### Files Modified
| File | Change |
|------|--------|
| `tracking/game_tracker.py` | Integrates `SessionState`; scans usernames every frame; post-processes action usernames; new hand_id format; passes `scan_ms` to `on_hand_start` |
| `tracking/logger.py` | `on_hand_start` displays `scan=XXXms` |
| `tracking/table_scanner.py` | `_determine_status` uses stack presence to detect occupied seats with blank OCR (label showing) — fixes dealer detection when dealer seat name region is obscured |

### Username persistence
`SessionState` maintains `{seat → username}` across all hands for a full session:
- `update_usernames(img, cfg)` runs every frame via `RegionCache` — only OCR-reads a region when pixels change
- OCR results are cleaned with `_CLEAN_RE` (strips `'`, `/`, `|`, `:`, `;`, `-`, etc.) before blocklist matching
- `_LABEL_BLOCKLIST` filters known action/status strings (`check`, `bet`, `raise`, `call`, `fold`, `post bb`, `post sb`, `all in`, `sitting out`, etc.)
- Only updates stored username if the new clean value differs from stored (no redundant writes)
- Seat cleared on EMPTY detection; kept on SITTING_OUT
- `table_state` player usernames patched from session state after each `scanner.scan()` — logger always shows clean names

### Hand ID format
Changed from `hand_0001` to `hand_{session_id}_{counter}` — e.g. `hand_20260312_160905_0001`. Globally unique across sessions.

### JSONL session log
`session_logs/session_{session_id}.jsonl`:
- Line 1: session header `{"type": "session", "session_id": ..., "table": ..., "started_at": ...}`
- Lines 2+: one completed hand per line, appended immediately on `on_hand_complete` (no data loss on crash)
- Hand record includes: hand_id, timestamps, BB, hero cards, community cards, positions, all actions with amounts/BB/dollars/street_totals/stacks

### Dealer detection fix
`_determine_status` previously returned `EMPTY` when OCR read blank from a name region (label showing that Tesseract couldn't parse). If the dealer sat at that seat, `assign_positions` would exclude them from `active` and return `{}` (no positions). Fix: blank username + numeric stack → `ACTIVE` (player present with label obscuring name).

### Session boundary
One `GameTracker` instance = one session. New PokerStars table = new window = new process = new session. No detection needed.

---

## Pending / Next Phases

| Phase | Goal |
|-------|------|
| 6 | Data aggregation & output (hand history JSON, WebSocket/live feed, GTO integration) |
| Improvement | Inferred CHECK detection: derive implied CHECKs from action ordering when a seat is skipped between `_last_actor` and the current actor. Removed in favour of direct blue-label detection; add back if gaps in CHECK coverage are observed in live sessions. |
| Improvement | Hand end detection: tail PokerStars hand history file for ground-truth result (winner, pot size, showdown cards) |
| Validation | Collect Phase 3–5 test images from live session → run `test_game_tracker.py` |
