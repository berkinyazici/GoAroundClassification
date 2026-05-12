# BBL514E Final Presentation — Slide Script
# Go-Around Classification Using ADS-B and METAR Data
# Furkan Güney (704241023) · Alper Berkin Yazıcı (704241020)

---

## SLIDE 1 — Title
**[~15 seconds]**

**Title:** Go-Around Classification Using ADS-B and METAR Data

**Subtitle:** Rare-event classification with engineered aviation/weather features

**Authors:** Furkan Güney · Alper Berkin Yazıcı

**What to say:**
> "Our project predicts go-around risk from public ADS-B-derived landing data and METAR weather observations. The main challenge is that go-arounds are extremely rare."

---

## SLIDE 2 — What is a Go-Around?
**[~35 seconds]**

**Bullets:**
- Aborted landing followed by climb-out and another approach
- Safety-preserving, but increases pilot/controller workload
- Disrupts arrival sequence and runway efficiency
- Rare event: current split has about **0.34-0.35 % positives**

**Key message:**
> This is a rare-event detection problem, not a standard balanced classifier.

**What to say:**
> "Accuracy is not enough here. A classifier that always predicts normal landing can already exceed 99 percent accuracy, so we focus on precision-recall behavior."

---

## SLIDE 3 — Dataset and Split
**[~45 seconds]**

**Source:** Large Landing Trajectory Dataset for Go-Around Analysis

**Available data:**
- ADS-B-derived landing context: airport, runway, aircraft type, glide slope, runway length, time
- METAR weather: wind, gust, visibility, temperature, pressure, weather codes
- Label: go-around vs normal landing

**Current processed temporal split:**

| Split | Rows | Go-arounds | Positive rate |
|---|---:|---:|---:|
| Train | 283,355 | 993 | 0.350 % |
| Validation | 82,484 | 280 | 0.339 % |
| Test | 82,474 | 283 | 0.343 % |

**What to say:**
> "The full source dataset has around 9 million landings, but our reproducible experiment uses this temporal working split. We keep validation and test untouched and tune only on validation."

---

## SLIDE 4 — Leakage Control
**[~35 seconds]**

**Removed from model inputs:**
```text
n_approaches
n_rwy_approached
icao24
callsign
registration
raw time
target / has_ga
```

**Why?**
- `n_approaches` is post-hoc: a go-around creates another approach
- aircraft IDs/callsigns can create memorization
- raw label fields directly leak the answer

**What to say:**
> "Earlier perfect-looking metrics were suspicious. The biggest issue was post-hoc information such as number of approaches. We removed these to make the evaluation honest."

---

## SLIDE 5 — Feature Sets
**[~50 seconds]**

**1. Context Only**
- airport/runway/aircraft/time
- glide slope angle and runway length

**2. Context + METAR**
- context features
- wind, gust, visibility, pressure, temperature, weather codes

**3. Context + METAR + Engineered**
- runway-relative wind:
  - headwind
  - tailwind
  - crosswind
- weather severity:
  - gust spread
  - low visibility flags
  - strong wind/crosswind flags
  - adverse weather flag
- train-only risk encodings:
  - airport risk
  - runway risk
  - aircraft type risk

**What to say:**
> "The engineered set uses only existing columns. No new dataset was needed. For risk encodings, we calculate rates only from the training split to avoid leakage."

---

## SLIDE 6 — Imbalance Handling
**[~40 seconds]**

**Problem:** positive rate is only ~0.34 %

**What we did:**
- keep all positive samples
- downsample negatives to **10:1** in training
- use class weights where supported
- LightGBM uses `scale_pos_weight`
- tune threshold on validation using **F2-score**
- select models primarily by validation **PR-AUC**

**What to say:**
> "F2 gives recall more weight than precision, which fits a safety-oriented detection problem. But the final model is still selected by PR-AUC because ranking rare positives is the central challenge."

---

## SLIDE 7 — Model Comparison
**[~60 seconds]**

**Latest results, ordered by validation PR-AUC:**

| Model | Feature Set | Val PR-AUC | Test ROC-AUC | Test PR-AUC | Test Recall |
|---|---|---:|---:|---:|---:|
| **MLP** | **context+METAR** | **0.0098** | 0.583 | 0.0058 | 0.0636 |
| LightGBM | engineered | 0.0093 | 0.604 | 0.0056 | 0.0565 |
| LDA | context+METAR | 0.0082 | 0.624 | 0.0064 | 0.0495 |
| LDA | engineered | 0.0081 | **0.652** | **0.0067** | **0.0954** |
| Logistic Reg. | context+METAR | 0.0081 | 0.620 | 0.0066 | 0.0848 |
| Random Forest | engineered | 0.0078 | 0.636 | 0.0063 | 0.0601 |

**What to say:**
> "The best validation PR-AUC is still MLP with context plus METAR. Engineered features improved ROC-AUC for several models, especially LDA and Logistic Regression, but did not fully solve the rare-event separation problem."

---

## SLIDE 8 — Final Model and Confusion Matrix
**[~45 seconds]**

**Final model:**
```text
MLP + context_metar
threshold = 0.2116
```

**Test metrics:**

| Metric | Value |
|---|---:|
| Accuracy | 0.9729 |
| Precision | 0.0091 |
| Recall | 0.0636 |
| F1 | 0.0159 |
| ROC-AUC | 0.5831 |
| PR-AUC | 0.0058 |

**Confusion matrix:**

|  | Pred Normal | Pred Go-Around |
|---|---:|---:|
| Actual Normal | 80,224 | 1,967 |
| Actual Go-Around | 265 | 18 |

**What to say:**
> "The model detects 18 of 283 go-arounds. Precision is low because even a small false-positive rate creates many false alarms when positives are extremely rare."

---

## SLIDE 9 — Feature Importance
**[~45 seconds]**

**Method:** permutation importance on average precision for the final MLP model

**Top features:**

| Rank | Feature | Relative impact |
|---:|---|---:|
| 1 | wind_gust_knts | 1.000 |
| 2 | weather_desc | 0.845 |
| 3 | weather_other | 0.625 |
| 4 | weather_precipitation | 0.555 |
| 5 | operator_region | 0.541 |
| 6 | wind_speed_knts | 0.392 |
| 7 | weather_intensity | 0.193 |
| 8 | icaoaircrafttype | 0.165 |

**What to say:**
> "Because MLP has no native tree importance, we use permutation importance. This tells us how much average precision drops when each feature is shuffled."

---

## SLIDE 10 — Explainable Web Interface
**[~45 seconds]**

**New interface features:**
- Inputs ordered by final-model feature importance
- Each input shows rank and relative impact bar
- Derived features shown read-only
- Derived feature dependencies are displayed
- Backend recalculates derived values when connected inputs change
- Local what-if panel after prediction:
  - e.g. `wind_gust_knts +5` changes go-around probability by X
  - `visibility_m -1000` changes probability by Y

**Endpoints:**
```text
/predict
/derived-features
/feature-importance
/sensitivity
```

**What to say:**
> "The demo is not just a prediction form anymore. It shows which inputs matter globally and how changing current numeric values changes the local probability."

---

## SLIDE 11 — System Architecture
**[~35 seconds]**

```text
Browser UI
   |
   | /predict, /derived-features, /feature-importance, /sensitivity
   v
FastAPI backend
   |
   | loads
   v
final_model.joblib + feature_schema.json
   |
   v
Prediction + explanation panels
```

**What to say:**
> "The backend centralizes all feature engineering. The UI never hand-computes model inputs; it asks the backend, so displayed derived values match prediction-time values."

---

## SLIDE 12 — Conclusion
**[~35 seconds]**

**Findings:**
- Public ADS-B-derived + METAR data provides weak but measurable signal
- Accuracy is misleading due to extreme imbalance
- PR-AUC and confusion matrix are more honest
- Engineered runway/weather/risk features improve some models, especially ROC-AUC
- Final precision/recall remain low, showing the limits of landing-level aggregate features

**Future work:**
- extract raw ADS-B trajectory time-series features
- speed/descent/altitude stability during final approach
- runway alignment and lateral deviation over time
- airport-specific calibration
- sequence models such as LSTM/Transformer

**Final line:**
> "The current system is a transparent baseline; the next leap requires raw approach trajectory dynamics."

---

# Demo Script

## 1. Start the app

```bash
/Users/berkinyazici/Desktop/ITU\ CS/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## 2. Show ranked inputs

Point out:
- `wind_gust_knts` is rank #1
- weather code fields are near the top
- lower-ranked fields are still available but visually lower priority

## 3. Fill sample input

Click **Fill Sample Input**.

Explain:
> "Derived features are computed by the backend. The user cannot edit them directly."

## 4. Modify important fields

Change:

```text
wind_gust_knts: 18 -> 35
wind_speed_knts: 12 -> 25
visibility_m: 8000 -> 1000
weather_precipitation: RA
weather_obscuration: FG
```

Show:
- derived gust spread changes
- low visibility flag changes
- predicted probability changes
- local sensitivity panel shows plus/minus effects

## 5. Close

Say:
> "This demonstrates the full loop: leakage-controlled features, imbalance-aware modeling, ranked feature importance, backend-derived features, and local what-if sensitivity."
