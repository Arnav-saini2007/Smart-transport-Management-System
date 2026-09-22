# ITMS — AI-Powered VANET Misbehavior Detection

An intelligent transport security system that detects malicious/misbehaving vehicles in a **VANET (Vehicular Ad-hoc Network)** in real time, using causal machine-learning features computed from V2X Basic Safety Messages (BSMs), and explains every flagged event in plain language using a local LLM (Gemma 3, via Ollama).

The system also includes a lightweight, rule-based **driving-safety layer** (crash / near-miss / harsh-braking / harsh-cornering detection) that runs alongside the security classifier.

> This README covers the detection pipeline (data, models, RSU simulation, inference API, driving-safety rules). The visualization dashboard is intentionally excluded here.

---

## How it works

```
┌─────────────┐     ┌──────────────┐     ┌───────────────────┐     ┌──────────────┐
│  data.csv   │────▶│  rsu_node.py │────▶│ model_service_v2.py│────▶│ alerts_log   │
│ (simulated  │     │ (RSU edge    │     │ (Flask API +       │     │ .jsonl       │
│  BSM stream)│     │  node)       │     │  LightGBM + Gemma) │     │              │
└─────────────┘     └──────┬───────┘     └────────────────────┘     └──────────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ driving/         │
                   │ crash_rules.py   │  (local, rule-based —
                   │ kinematic_       │   crash / near-miss /
                   │ features.py      │   harsh braking & cornering)
                   └──────────────────┘
```

1. **`rsu_node.py`** replays a BSM dataset (`data.csv`) in chronological order, simulating a Roadside Unit (RSU) receiving live V2X messages.
2. Each message is sent to a **Flask inference API** (`model_service_v2.py`), which computes causal, deployment-safe features and classifies the message as `Normal` or `Attack`.
3. Flagged messages are explained in natural language by **`gemma_layer.py`**, which prompts a local Gemma 3 model via [Ollama](https://ollama.com) — grounded in the actual signal that triggered the flag (e.g. message flooding vs. position spoofing), not a generic template.
4. In parallel, **`driving/crash_rules.py`** evaluates each message against physics-based thresholds (deceleration, jerk, yaw rate, time-to-collision) to catch instantaneous driving-safety events.
5. Everything is appended to `alerts_log.jsonl`, a shared structured log.

---

## Key design decisions

| Decision | Why |
|---|---|
| **Causal features only** | Every feature (`msg_count_so_far`, `time_since_last_msg`, `msg_rate_1s`, rolling stats, etc.) is computable message-by-message from *past* data only — no features that peek into the future. This closes the train/serve gap so the trained model behaves identically in live deployment. |
| **Vehicle-disjoint train/test split** | Uses `GroupShuffleSplit` on `sender` so no vehicle appears in both train and test. A random row split leaks vehicle identity and inflates accuracy; this gives an honest estimate of generalization to *unseen* vehicles. |
| **Balanced training** | Normal/Attack classes are balanced by undersampling, and attack subtypes are individually capped so high-volume subtypes (e.g. DoS variants) don't drown out rare ones (e.g. replay attacks). |
| **Binary + multi-class hybrid** | A binary LightGBM model catches *any* anomaly; a multi-class model refines detection for historically hard subtypes (constant/random position offset, replay, DoS). |
| **Per-subtype threshold tuning** | A single global 0.5 threshold misses subtle attacks like replay and DoS. Thresholds are tuned per attack subtype on a validation split of vehicles that is disjoint from both training *and* final evaluation vehicles (`fix_threshold_leak.py`). |
| **Rule-based driving-safety layer** | Crash/near-miss/harsh-event detection uses explainable physics thresholds rather than ML — these are instantaneous, well-understood physical signatures where a threshold beats a learned model. |
| **Local LLM explanations** | Gemma 3 (via Ollama) runs locally — no API costs, no data leaving the network — and is prompted with the actual dominant signal (frequency vs. position vs. speed anomaly) so explanations are grounded rather than hallucinated. |

---

## Project structure

```
.
├── data.csv                       # Input BSM dataset (VeReMi-style schema; gitignored)
├── rsu_node.py                    # RSU edge simulator: streams BSMs to the model API
├── model_service_v2.py            # Flask inference API (/classify) — causal features + Gemma
├── gemma_layer.py                 # Gemma 3 prompt engineering for anomaly explanations
├── driving/
│   ├── kinematic_features.py      # Causal accel/jerk/yaw-rate/TTC feature computation
│   └── crash_rules.py             # Rule-based CRASH / NEAR_MISS / HARSH_BRAKE / HARSH_CORNER detector
├── train_model.py                 # Baseline RandomForest, causal features only
├── train_model_enhanced.py        # LightGBM binary + multi-class, sequence features, per-class thresholds
├── train_model_full.py            # Full-dataset training run (RandomForest)
├── train_model_full_v2.py         # Full-dataset training, per-attack-subtype balanced
├── fix_threshold_leak.py          # Re-tunes thresholds with a vehicle-disjoint val/eval split
├── convert_for_dashboard.py       # Converts alerts_log.jsonl → data/predictions.csv
├── check.py                       # Quick dataset sanity check (columns/shape)
└── requirements.txt
```

---

## Data

The pipeline expects a VeReMi Extension–style dataset with columns for position, speed, acceleration, and heading (both ground-truth and noisy/received `_n` variants), plus a `sender` (vehicle id), `sendTime`, and a `class` label (`0` = normal, `1–19` = distinct misbehavior/attack subtypes).

- Training scripts expect the full dataset at a local path (edit `DATA_PATH` in each script).
- Live simulation (`rsu_node.py`) expects a `data.csv` in the working directory.

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install lightgbm flask requests ollama joblib fpdf2  # not all pinned in requirements.txt

# 2. Pull the local LLM used for explanations
ollama pull gemma3:4b
```

### Train a model

```bash
# Baseline (RandomForest, causal features)
python train_model.py

# Enhanced (LightGBM binary + multi-class, sequence features, per-class thresholds)
python train_model_enhanced.py

# Full dataset (3M+ rows)
python train_model_full_v2.py

# Re-check threshold tuning for leakage (vehicle-disjoint val/eval)
python fix_threshold_leak.py
```

Each script saves its model artifacts (`.pkl`) and a training summary log.

### Run live detection

```bash
# Terminal 1 — start the inference API
python model_service_v2.py

# Terminal 2 — start the RSU simulator (streams data.csv to the API)
python rsu_node.py
```

Detections and driving-safety events are appended to `alerts_log.jsonl` as they occur.

---

## Requirements

See `requirements.txt` for pinned versions. Core dependencies not listed there but required by the scripts above:

- `lightgbm` — enhanced training pipeline
- `flask` — inference API
- `requests` — RSU → API communication
- `ollama` — local Gemma 3 explanations
- `joblib` — model persistence
- `fpdf2` — PDF report export (used by the export utilities)

---

## License

Add your license of choice here.
