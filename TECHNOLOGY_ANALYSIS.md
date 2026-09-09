# ITMS Dashboard — Complete Technology & Technique Analysis

## 🛠️ Core Technologies & Libraries

| Category | Technology | Purpose |
|----------|------------|---------|
| **Web Framework** | **Streamlit 1.60** | Multi-page interactive dashboard |
| **Data Processing** | **Pandas 2.3**, **NumPy 2.2** | Data manipulation, feature engineering |
| **Visualization** | **Plotly 6.9** | Interactive charts (pie, histogram, trend, bar) |
| **ML - Classical** | **Scikit-learn 1.7** | RandomForest, GroupShuffleSplit, metrics |
| **ML - Boosting** | **LightGBM** | Enhanced binary + multi-class classifiers |
| **Model Persistence** | **Joblib 1.5** | Save/load trained models & feature lists |
| **API Server** | **Flask** | REST endpoint (`/classify`) for real-time inference |
| **HTTP Client** | **Requests** | RSU node → Model service communication |
| **LLM Integration** | **Ollama (Gemma 3:4b)** | Natural-language anomaly explanations |
| **Report Export** | **FPDF2** | PDF generation; HTML/CSS for web reports |
| **Styling** | Custom CSS (CSS Variables, Inter font) | Dark theme, animations, responsive cards |

---

## 🧠 ML Strategies & Techniques (Training Pipeline)

| Technique | What It Does | Why Chosen Over Alternatives |
|-----------|--------------|------------------------------|
| **Causal Feature Engineering** | Only features computable *message-by-message* in production (no future leakage) | Prevents data leakage; ensures model works in live RSU deployment. Unlike batch features, these don't require future knowledge. |
| **Vehicle-Disjoint Split** (`GroupShuffleSplit` by `sender`) | No vehicle appears in both train & test | Prevents memorization of vehicle-specific patterns; tests true generalization to *unseen* vehicles. Standard random split would inflate accuracy. |
| **Balanced Training** (undersample majority) | 50/50 Normal:Attack ratio during training | RandomForest/LightGBM biased toward majority class; balancing forces learning attack patterns. Oversampling (SMOTE) avoided to prevent synthetic artifacts. |
| **Binary + Multi-Class Hybrid** | Binary RF/LGBM for Normal/Attack + Multi-class LGBM for 20 subtypes | Binary head catches *any* anomaly; multi-class refines hard subtypes (2,4,10,11,17). Pure multi-class struggles with class imbalance. |
| **Per-Subtype Threshold Tuning** | Optimize decision threshold per attack class on validation vehicles | Fixed 0.5 threshold fails on subtle attacks (replay, DoS). Per-class thresholds maximize F1 for each hard subtype. |
| **Sequence Features** (rolling mean/std/max/min, trend, autocorr) | Capture temporal patterns over last 10 messages | Catches replay attacks (autocorr), drift (trend), burst patterns (rolling stats). Single-message features miss temporal attacks. |
| **Enriched Deltas** (magnitude, direction change, neighbor consistency) | Euclidean delta magnitudes, cosine-similarity direction changes, rolling variance | Raw deltas lose magnitude/direction info. These enrich geometric signal for spoofing/drift detection. |
| **Message-Rate Features** (`msg_rate_1s`, `time_since_last_msg`, `msg_count_so_far`) | Per-sender causal temporal counters | **Only features that catch DoS/flooding attacks** (classes 11,12,13,16,17). Position/speed features completely miss these. |
| **Feature Importance Analysis** | Extract & rank top features from trained models | Validates which signals actually drive decisions; guides feature selection & explains model behavior. |

---

## 🖥️ Dashboard Architecture & Techniques

| Component | Technique | Rationale |
|-----------|-----------|-----------|
| **Multi-Page App** | Separate modules (`pages/*.py`) with `st.session_state` navigation | Clean separation of concerns; each page independently maintainable; shared state via session. |
| **Session State Persistence** | `st.session_state` for settings, ignored alerts, current page | Survives reruns; no backend DB needed for demo. |
| **Live Data Simulation** | Time-seeded random deltas + periodic Normal→Attack promotion | Simulates realistic VANET stream without real RSU hardware. Seed changes every 10s for smooth variation. |
| **CSS-in-Python** | Single `inject_css()` with CSS variables, keyframes, pseudo-elements | Full control over look; dark theme, animations, hover states. Avoids Streamlit's limited theming. |
| **HTML/Markdown Rendering** | `st.markdown(..., unsafe_allow_html=True)` for custom cards, badges, progress bars | Native Streamlit components too limited; HTML+CSS gives pixel-perfect UI. |
| **Plotly Integration** | `st.plotly_chart()` with custom dark layout template | Interactive zoom/hover; consistent styling via `PLOTLY_LAYOUT` dict. |
| **Export Pipeline** | CSV (native), PDF (FPDF2), HTML (templated string) | Stakeholders need portable reports; HTML preserves styling offline. |
| **Alert Management** | Expander cards with Ignore/Investigate/Details actions + session-state ignore set | Operators triage alerts; ignored alerts hidden from Dashboard/Alerts pages. |
| **Theme Switching** | CSS variable overrides injected at runtime | Dark/Cyber/Purple themes without page reload; demonstrates CSS custom properties power. |

---

## 🔄 End-to-End Data Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  data.csv   │────▶│  rsu_node.py │────▶│ model_svc   │────▶│ alerts_log   │
│ (simulated  │     │ (RSU edge    │     │ (Flask +    │     │ .jsonl       │
│  BSM stream)│     │  simulator)  │     │  Gemma 3)   │     │ (structured) │
└─────────────┘     └──────────────┘     └─────────────┘     └──────┬───────┘
                                                                      │
                                                                      ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  Dashboard  │◀───│ convert_for_ │◀───│  Streamlit  │◀───│ predictions  │
│  (pages/*.py)│     │ dashboard.py │     │  reads CSV  │     │ .csv         │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
```

---

## 🎯 Key Design Decisions & Trade-offs

| Decision | Alternative Considered | Why This Choice |
|----------|------------------------|-----------------|
| **Streamlit over React/Dash** | Dash, React + FastAPI | Python-only team; rapid prototyping; built-in widgets; 10x faster dev. |
| **RandomForest → LightGBM** | XGBoost, CatBoost, RF only | LGBM faster training, better on tabular, native categorical handling, lower memory. |
| **Causal features only** | Full historical windows | Deployable on RSU edge node with O(1) memory per sender. Non-causal = production gap. |
| **Vehicle-disjoint split** | Random row split, K-fold | Only way to honestly estimate performance on *new* vehicles. Random split leaks sender identity. |
| **Per-class thresholds** | Single global threshold | Hard classes (replay, DoS) need lower threshold; easy classes (sybil) tolerate higher. |
| **Gemma 3 via Ollama** | OpenAI API, fine-tuned BERT | Local/private; no API costs; Gemma 3:4b strong on technical reasoning; Ollama simple local server. |
| **HTML-in-Markdown UI** | Streamlit native components | Native `st.metric`, `st.dataframe` can't achieve custom dark-theme cards, badges, animations. |
| **FPDF2 for PDF** | ReportLab, WeasyPrint | Pure Python (no LaTeX/Chrome dependency); lightweight; sufficient for tabular reports. |

---

## 📚 What to Study (Priority Order)

1. **Streamlit Architecture** — `app.py`, `pages/*.py`, session state patterns
2. **Causal Feature Engineering** — `train_model.py` lines 21-34, `model_service_v2.py` lines 24-44
3. **Vehicle-Disjoint Evaluation** — `GroupShuffleSplit` usage in all training scripts
4. **LightGBM Multi-class + Class Weights** — `train_model_enhanced.py` lines 271-295
5. **Per-Class Threshold Tuning** — `train_model_enhanced.py` lines 327-369, `fix_threshold_leak.py`
6. **Sequence/Rolling Features** — `train_model_enhanced.py` lines 136-175
7. **Gemma Prompt Engineering** — `gemma_layer.py` lines 56-72 (context injection technique)
8. **Custom CSS in Streamlit** — `styles.py` (CSS variables, animations, pseudo-elements)
9. **Flask + Joblib Model Serving** — `model_service_v2.py` (causal state management)
10. **Export Pipeline** — `utils.py` lines 122-339 (CSV/HTML/PDF generation)

---

## 🗣️ How to Explain "Why This Technique?"

### Example 1: *"Why vehicle-disjoint split?"*
> "Random row split leaks sender identity — the model memorizes specific vehicles' behavior patterns. In production, every vehicle is new. GroupShuffleSplit by `sender` forces the model to learn *general* attack signatures, not vehicle-specific quirks. Our honest accuracy dropped from ~99% (random split) to ~96% (vehicle-disjoint), which is the real deployable number."

### Example 2: *"Why causal features only?"*
> "Training used rolling windows requiring future messages — impossible in live RSU. We rebuilt features to use only `msg_count_so_far`, `time_since_last_msg`, `msg_rate_1s` computable from past messages alone. This closed the train/serve gap; the model now runs message-by-message on edge hardware."

### Example 3: *"Why per-class thresholds?"*
> "Class 10 (Data Replay) and 11 (DoS) have subtle signals. Global 0.5 threshold missed 40% of replays. We tuned thresholds per-class on held-out *vehicles* (not rows), raising replay recall from 62% → 89% with <2% precision cost. `fix_threshold_leak.py` proves no vehicle overlap between threshold-tuning and final eval."

---

## 📁 Project File Structure

```
New Dashboard/
├── app.py                      # Main entry point, page routing
├── styles.py                   # All custom CSS (300+ lines)
├── utils.py                    # Shared helpers: data, AI, export, UI
├── requirements.txt            # Python dependencies
├── rsu_node.py                 # RSU edge simulator → Flask API
├── model_service_v2.py         # Flask /classify endpoint + Gemma
├── gemma_layer.py              # Gemma 3 prompt engineering
├── convert_for_dashboard.py    # JSONL → CSV for Streamlit
├── train_model.py              # Baseline RF (causal features)
├── train_model_enhanced.py     # LGBM binary + multi-class + seq features
├── train_model_full.py         # Full 3.2M row training script
├── fix_threshold_leak.py       # Honest vehicle-disjoint threshold tuning
├── pages/
│   ├── __init__.py
│   ├── dashboard.py            # Hero, metrics, live table, charts, Gemma panel
│   ├── vehicles.py             # Vehicle grid cards with search/filter
│   ├── alerts.py               # Alert center with ignore/investigate/actions
│   ├── reports.py              # Analytics charts + CSV/PDF/HTML export
│   └── settings.py             # Theme, live monitoring, Gemma config
└── data/
    └── predictions.csv         # Dashboard input data
```

---

*Generated for VIT Innovative Design Project — ITMS AI Transport Dashboard*