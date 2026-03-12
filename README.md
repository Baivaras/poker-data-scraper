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

## Capturing screenshots for testing

Press **SPACE** while `main.py` is running to save a timestamped screenshot to `captures/`.
These screenshots are the input for both the labeller and the card detection test.

---

## Card detection — label and test

### Step 1 — Label captured images
```bash
python3 label_cards.py
python3 label_cards.py --images captures/ --labels label_cards.json
python3 label_cards.py --redo   # re-label already labelled images
```

| Argument | Default | Description |
|---|---|---|
| `--images` | `captures/` | Directory with PNG screenshots |
| `--labels` | `label_cards.json` | Output JSON file for labels |
| `--config` | `regions.json` | Regions config path |
| `--redo` | off | Re-label images that already have an entry |
| `--limit` | `0` (all) | Stop after N images |

Opens each screenshot in the system viewer, runs detection, and prompts you to confirm or correct.
Labels are saved after every image so progress is never lost — resume any time.

**Controls at each prompt:**

| Key | Action |
|---|---|
| Enter | Accept the detected value |
| `Ah Ks` | Override with these cards (space-separated) |
| `-` | Mark as not present / empty board |
| `s` | Skip this image (no label saved) |
| `b` | Go back to the previous image |
| `q` | Quit and save |

Labels are stored as JSON:
```json
{
  "cap_20260311_163551.png": {
    "hero": ["Kd", "2h"],
    "board": ["9d", "7c", "Tc", "Qc", "8h"]
  }
}
```

### Step 2 — Test detection accuracy
```bash
python3 test_cards.py
python3 test_cards.py --images captures/ --labels label_cards.json
python3 test_cards.py --limit 20
```

| Argument | Default | Description |
|---|---|---|
| `--images` | `captures/` | Directory with PNG screenshots |
| `--labels` | `label_cards.json` | Label JSON from `label_cards.py` |
| `--config` | `regions.json` | Regions config path |
| `--limit` | `0` (all) | Stop after N images |

Runs both hero and board card detection on every image and prints a table with separate
**Hero?** and **Board?** pass/fail columns plus per-image confidence and latency.
Accuracy summary at the bottom shows correct/wrong/missed counts for each.

### Retraining the card model

After labelling new captures, retrain to include the new board crop distribution:
```bash
python3 NN/train_card_model.py
python3 NN/train_card_model.py --epochs 40
python3 NN/train_card_model.py --hero-only   # skip board crops
```

| Argument | Default | Description |
|---|---|---|
| `--epochs` | `60` | Number of training epochs |
| `--lr` | `3e-4` | Learning rate |
| `--hero-only` | off | Train on hero crops only (skip board crops) |
| `--no-pretrain` | off | Start from random weights instead of loading existing model |

Combines hero crops from `tests/` + `label_hands.txt` with board crops from
`captures/` + `label_cards.json`. Overwrites `NN/card_detector_nn.pth`.

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
├── label_hands.txt           # Hero card ground-truth labels (image → card1 card2)
├── label_cards.json          # Hero + board ground-truth labels (from label_cards.py)
├── label_cards.py            # Interactive labeller for hero + board cards
├── test_cards.py             # Hero + board card detection accuracy test
├── test_hero_cards.py        # Legacy hero-only accuracy test
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
│   ├── action_detector.py    #   Stack delta → PlayerAction events, CHECK inference, parallel OCR
│   ├── bb_detector.py        #   BB detection from window title or pot OCR
│   ├── board_detector.py     #   Community card detection (per-slot hash cache)
│   ├── region_cache.py       #   RegionCache — skip detection when region pixels unchanged
│   └── street_detector.py    #   Board slot variance → Street (PREFLOP/FLOP/TURN/RIVER)
│
├── tracking/                 # State machines and orchestrators
│   ├── table_scanner.py      #   Phase 2: single-frame snapshot → TableState
│   ├── game_tracker.py       #   Phase 3: unified poller, full HandState tracking
│   └── logger.py             #   PokerLogger — terminal + file event logging
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
    ├── nn_card_reader.py         # MobileNetV2 dual-head card classifier
    ├── train_card_model.py       # Retrain on hero + board crops
    ├── train_corner_model.py     # Train corner-crop model (legacy)
    ├── card_detector_nn.pth      # Trained weights — hero + board crops (rank + suit)
    └── corner_card_nn.pth        # Trained weights — corner crop variant
```

---

## Detection pipeline & performance

Each live frame goes through a fixed sequence of detectors. The table below shows
which ones run, when they are skipped, and what caching applies.

| Detector | Runs when | Skipped when | Caching |
|---|---|---|---|
| Street detector (board variance) | Every frame | — | Per-slot region hash — skips numpy variance if slot pixels unchanged |
| Hero card detector (NN) | Every frame | Variance gate fails | Full-crop perceptual hash — skips NN inference when pixels unchanged |
| Board card detector (NN) | Street transition only | Street didn't change | Per-slot perceptual hash — skips NN when slot unchanged |
| Opponent fold detection (variance) | Every frame | Cards region unchanged | Region hash cache — skips variance check on static frames |
| Opponent stack OCR | Every frame | — | **Not cached** — always re-read (primary action signal) |
| Opponent action detection | Every frame | — | — |
| Action button visibility (saturation) | Every frame | Region unchanged | Region hash cache — skips saturation scan when pixels unchanged |
| Hero action detection (stack delta) | Buttons just disappeared | — | — |

### Region hash cache (`detection/region_cache.py`)

`RegionCache` stores an 8×8 grayscale thumbnail hash per region. `cache.changed(img, region)`
returns `True` only when the pixel content differs from the previous call. Stack and bet
regions are excluded — they must always be re-read as they are the primary action signal.

---

## Adding a new poker room

1. Drop `images/{Room}_dealer_button.png` (cropped screenshot of the dealer chip).
2. Re-calibrate regions with `tools/mark_regions.py` + `tools/mark_dealer_regions.py`.
3. Pass `--room Room` when running `main.py` or any test script.
