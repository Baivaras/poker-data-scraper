# Poker Tracker

Real-time PokerStars table tracker — screen capture + OCR + neural net card detection.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install tesseract   # macOS, required for OCR
```

---

## Calibration (one-time per table layout)

### Step 1 — Mark all regions
```bash
python3 tools/mark_regions.py <screenshot.png>
python3 tools/mark_regions.py <screenshot.png> --regions regions.json
```

| Argument | Default | Description |
|---|---|---|
| `image` | *(required)* | Screenshot PNG to use as background |
| `--regions` | `regions.json` | Output path for region coordinates |

Draws bounding boxes for hero cards, hero name/stack, action buttons, pot,
board slots, and all 5 opponent seat regions. Saves to `regions.json`.

### Step 2 — Mark dealer button regions
```bash
python3 tools/mark_dealer_regions.py <screenshot.png>
python3 tools/mark_dealer_regions.py <screenshot.png> --regions regions.json
```

| Argument | Default | Description |
|---|---|---|
| `image` | *(required)* | Screenshot PNG to label on |
| `--regions` | `regions.json` | Output path for region coordinates |

Draws the small area where the dealer chip appears for each seat + hero. Updates `regions.json`.

---

## Running live

```bash
python3 main.py
python3 main.py --room PS --config regions.json --verbose
python3 main.py --screenshots captures/ --debug
```

| Argument | Default | Description |
|---|---|---|
| `--config` | `regions.json` | Region coordinates config file |
| `--room` | `PS` | Poker room identifier (used for dealer template matching) |
| `--screenshots` | `captures/` | Directory for SPACE-key screenshots |
| `--debug` | off | Enable action-detector debug output (stack deltas, noise floor, debounce) |
| `--verbose` | off | Show stack before/after and delta next to each action in the log |

When `--debug` or `--verbose` is active, all output (stdout + stderr) is also written to
`logs/tracker_YYYYMMDD_HHMMSS.log` with ANSI colour codes stripped.

Press **SPACE** at any time while running to save a timestamped screenshot to the screenshots directory.

---

## Tests

### Hero card accuracy
```bash
python3 test_hero_cards.py
python3 test_hero_cards.py --tests tests/ --labels label_hands.txt --config regions.json --limit 50
```

| Argument | Default | Description |
|---|---|---|
| `--tests` | `tests/` | Directory with PNG test images |
| `--labels` | `label_hands.txt` | Ground-truth label file |
| `--config` | `regions.json` | Regions config path |
| `--limit` | `0` (all) | Stop after N images |

### Table state detection
```bash
python3 test_table_state.py --mode print --limit 30
python3 test_table_state.py --mode label
python3 test_table_state.py --mode test
```

| Argument | Default | Description |
|---|---|---|
| `--mode` | `print` | `print` — inspect output · `label` — save as ground truth · `test` — regression against labels |
| `--tests` | `tests/` | Directory with PNG test images |
| `--config` | `regions.json` | Regions config path |
| `--room` | `PS` | Poker room identifier |
| `--limit` | `0` (all) | Limit number of images (print mode only) |

### Full game tracker replay
```bash
python3 test_game_tracker.py
python3 test_game_tracker.py --tests tests_phase3/ --limit 100 --json hands.json
```

| Argument | Default | Description |
|---|---|---|
| `--tests` | `tests/` | Directory with sequentially-named PNG frames |
| `--config` | `regions.json` | Regions config path |
| `--room` | `PS` | Poker room identifier |
| `--limit` | `0` (all) | Stop after N images |
| `--json` | *(none)* | Write completed hands as JSON to this file |

---

## Project structure

```
poker-tracker/
├── main.py                   # Live entry point
├── regions.json              # Calibrated fractional region coordinates
├── requirements.txt          # Python dependencies
├── label_hands.txt           # Phase 1 ground-truth labels (card per image)
├── test_hero_cards.py        # Phase 1 accuracy test
├── test_table_state.py       # Phase 2 print / label / test modes
├── test_game_tracker.py      # Phase 3 full pipeline replay test
│
├── core/                     # Shared data types and config
│   ├── models.py             #   Card, HeroCardsResult dataclasses
│   ├── config.py             #   TrackerConfig, RegionConfig, load_config()
│   ├── hand_state.py         #   HandState, PlayerAction, ActionRound, Street
│   └── table_state.py        #   PlayerState, TableState, assign_positions()
│
├── detection/                # Per-frame computer vision modules
│   ├── capture.py            #   macOS Quartz screenshot, get_pokerstars_window()
│   ├── card_detector.py      #   Variance gate → NN → HeroCardsResult
│   ├── dealer_detector.py    #   Template-match dealer chip across seat regions
│   ├── ocr.py                #   Tesseract wrappers: read_text(), read_number(), read_stack()
│   ├── action_detector.py    #   Stack delta → PlayerAction events, CHECK inference
│   ├── bb_detector.py        #   BB detection from window title or pot OCR
│   ├── board_detector.py     #   Community card detection
│   └── street_detector.py    #   Board slot variance → Street (PREFLOP/FLOP/TURN/RIVER)
│
├── tracking/                 # State machines and orchestrators
│   ├── table_scanner.py      #   Phase 2: single-frame snapshot → TableState
│   └── game_tracker.py       #   Phase 3: unified poller, full HandState tracking
│
├── tools/                    # Calibration GUI tools (run once per layout)
│   ├── mark_regions.py       #   Draw all seat/hero/board regions → regions.json
│   └── mark_dealer_regions.py#   Draw dealer button regions → regions.json
│
├── images/                   # Static assets
│   ├── PS_dealer_button.png  #   PokerStars dealer chip template
│   └── PS_empty_seat.png     #   Empty seat reference image
│
├── captures/                 # SPACE-key screenshots (timestamped)
├── logs/                     # Session log files (debug/verbose mode)
│
└── NN/
    ├── nn_card_reader.py     # MobileNetV2 dual-head card classifier
    ├── card_detector_nn.pth  # Pre-trained weights (rank + suit heads)
    └── corner_card_nn.pth    # Corner crop model
```

---

## Adding a new poker room

1. Drop `images/{Room}_dealer_button.png` (cropped screenshot of the dealer chip).
2. Re-calibrate regions with `tools/mark_regions.py` + `tools/mark_dealer_regions.py`.
3. Pass `--room Room` when running `main.py` or any test script.
