# BBL514E Final Presentation — Slide Script
# Go-Around Classification Using ADS-B and METAR Data
# Furkan Güney (704241023) · Alper Berkin Yazıcı (704241020)

---

## SLIDE 1 — Title Slide
**[~15 seconds]**

**Title:** Go-Around Classification Using ADS-B and METAR Data

**Subtitle:** BBL514E Pattern Recognition — Term Project

**Authors:** Furkan Güney (704241023) · Alper Berkin Yazıcı (704241020)

**Visual:** photo of an aircraft performing a go-around (nose up, gear still down over runway)
or a clean aviation-themed background

**What to say:**
> "Good [morning/afternoon]. Our project is about predicting go-arounds — aborted landing
> attempts — using open aviation and weather data. I'm [name], and this is [partner name]."

---

## SLIDE 2 — What is a Go-Around?
**[~45 seconds]**

**Title:** What is a Go-Around?

**Left side — image:** diagram showing approach path vs. go-around climb path

**Right side — bullet points:**
- Pilot aborts landing on final approach
- Climbs away and re-attempts
- **Safety-preserving** — but increases workload
- Rate: typically **0.1 – 2 %** of landings
- Our dataset: **≈ 0.37 %** → severe class imbalance

**Key message box:**
> Can we predict go-around risk from public data *before* touchdown?

**What to say:**
> "A go-around is when a pilot decides the landing isn't safe and climbs away to try again.
> It's the correct thing to do, but it disrupts traffic and increases controller workload.
> The challenge is they're very rare — under 0.4% of all landings — which makes this
> a difficult imbalanced classification problem."

---

## SLIDE 3 — Dataset
**[~45 seconds]**

**Title:** Dataset

**Large statistics (icons + numbers):**
| Icon | Stat |
|---|---|
| ✈️ | ~9 million landings |
| 🔴 | ~33,000 go-arounds (0.37 %) |
| 🏢 | 176 airports |
| 🌍 | 44 countries |
| 📅 | Full year 2019 |

**Source box:**
> Monstein et al. (2022) — Zenodo record 7148117
> `go_arounds_augmented.csv.gz`

**Temporal split diagram (3 boxes):**
```
Jan–Aug 2019          Sep–Oct 2019       Nov–Dec 2019
  TRAINING              VALIDATION           TEST
  5.66M rows             1.65M rows         1.65M rows
```

**What to say:**
> "We used a large public dataset from Zenodo covering all 2019 landings at 176 airports.
> We split it strictly by time — train on January through August, validate on September-October,
> test on November-December. This prevents any future data leaking into training."

---

## SLIDE 4 — Feature Engineering
**[~40 seconds]**

**Title:** Feature Engineering — Two Feature Sets

**Two columns:**

**Context Only (15 features)**
- Airport, runway, aircraft type
- Wake turbulence category
- Glide slope angle, runway length
- Month, day of week, hour (UTC)

**Context + METAR (27 features)**
= Context + Weather:
- Wind speed, direction, gusts
- Visibility, temperature, pressure
- Weather codes (fog, rain, snow...)

**Bottom callout (red warning box):**
⚠️ **Leakage removed:** `n_approaches` excluded — it counts total approaches per flight,
which post-hoc reveals the go-around.

**What to say:**
> "We defined two feature sets to run an ablation study. The context-only set uses
> airport, aircraft, and time information. The full set adds METAR weather observations.
> One important step was removing n_approaches from features — it counted how many
> approaches a flight made total, which directly leaks the answer."

---

## SLIDE 5 — Classifiers
**[~40 seconds]**

**Title:** Five Classifiers Compared

**Grid (2×3, last cell empty):**

| | | |
|---|---|---|
| **LDA** | **Logistic Regression** | **Random Forest** |
| Generative linear | Discriminative linear | 100-tree ensemble |
| Gaussian assumption | Weighted cross-entropy | Bootstrap + random split |
| **MLP** | **LightGBM** | |
| 64→32 hidden units | Gradient boosting | |
| ReLU + early stop | Histogram-based | |

**Bottom bar:**
All models: class-weighted training · threshold tuned on validation set for max F1

**What to say:**
> "We compared five classifiers — from the classical LDA baseline to neural and boosting models.
> All were trained with class weighting to handle the imbalance. After training, we tuned
> the decision threshold on the validation set to maximize F1 rather than using the default 0.5."

---

## SLIDE 6 — Results Table
**[~50 seconds]**

**Title:** Results — Test Set Performance

**Table (highlight MLP row in green):**

| Model | Feature Set | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|---|
| **🏆 MLP** | **context+METAR** | **0.685** | **0.0160** | **0.043** |
| LightGBM | context+METAR | 0.679 | 0.0134 | 0.030 |
| Random Forest | context+METAR | 0.690 | 0.0115 | 0.037 |
| Logistic Reg. | context+METAR | 0.691 | 0.0102 | 0.033 |
| LDA | context+METAR | 0.681 | 0.0100 | 0.034 |
| MLP | context only | 0.609 | 0.0123 | 0.037 |
| ... | context only | ~0.63 | ~0.008 | ~0.025 |

**Side callout box:**
> No-skill baseline PR-AUC ≈ 0.0035
> Best model: **4.6× above baseline**

**What to say:**
> "Here are the test set results. We rank by PR-AUC because it's the right metric for rare events —
> ROC-AUC can look good even on imbalanced problems. The MLP with full weather features
> came first by PR-AUC at 0.016 — about 4.6 times above the no-skill baseline."

---

## SLIDE 7 — Key Insight: Weather Matters
**[~30 seconds]**

**Title:** Key Finding — METAR Weather Features Consistently Help

**Bar chart (or table with arrows):**

| Model | Context Only PR-AUC | + METAR PR-AUC | Gain |
|---|---|---|---|
| MLP | 0.0123 | 0.0160 | **+30 %** |
| LightGBM | 0.0097 | 0.0134 | **+38 %** |
| Logistic Reg. | 0.0070 | 0.0102 | **+46 %** |
| LDA | 0.0075 | 0.0100 | **+33 %** |

**Visual:** insert `reports/figures/precision_recall_curve.png` or `roc_curve.png`

**What to say:**
> "Adding weather features improved every single model by 30 to 46 percent in PR-AUC.
> This confirms that visibility, wind, and weather codes carry real predictive signal
> that context features alone cannot capture."

---

## SLIDE 8 — Confusion Matrix
**[~30 seconds]**

**Title:** Best Model — Confusion Matrix (MLP, context+METAR)

**Insert:** `reports/figures/confusion_matrix.png`

**Below the figure:**
| | |
|---|---|
| Threshold | τ* = 0.131 (tuned on validation) |
| True Positives | 256 go-arounds detected |
| False Negatives | 5,525 go-arounds missed |
| False Positives | 5,870 false alarms |

**Bottom note:**
> Precision: 4.2 % · Recall: 4.4 % · Accuracy: 99.3 %
> (accuracy is misleading — model beats no-skill baseline by 4.6× on PR-AUC)

**What to say:**
> "The confusion matrix shows the inherent difficulty of the problem. Even the best model
> detects only about 4% of go-arounds. This is expected — go-arounds share almost identical
> observable conditions with the vast majority of normal landings."

---

## SLIDE 9 — System Architecture
**[~30 seconds]**

**Title:** Deployed System

**Diagram (left to right):**
```
[User Browser]
      │  HTTP POST /predict
      ▼
[HTML Interface]  ◄──── Jinja2 template
      │
[FastAPI Backend]  (port 8000)
      │
[MLP Model]  ◄──── final_model.joblib
      │
[Docker Container]  python:3.11-slim
```

**Right side bullets:**
- Single `docker compose up` to run
- `/predict` — JSON API endpoint
- `/` — HTML form interface
- `/health` — health check

**What to say:**
> "The system runs in a single Docker container. The FastAPI backend loads the trained
> model at startup and exposes both a JSON API and an HTML interface. Everything runs
> with one docker compose up command."

---

## SLIDE 10 — Conclusion
**[~25 seconds]**

**Title:** Conclusion

**Findings:**
✅ Go-around classification from public ADS-B + METAR data is feasible
✅ METAR weather features improve all models by 30–46% in PR-AUC
✅ Best model: MLP + context+METAR → ROC-AUC 0.685, PR-AUC 0.016 (4.6× baseline)
✅ Temporal train/val/test split → no leakage

**Limitations:**
- Only landing-level aggregate features (no per-second trajectory data)
- Low absolute precision/recall due to extreme class imbalance

**Future work:**
- Sequence models (LSTM) on per-second ADS-B trajectory
- Airport-specific fine-tuning

**Final line (bold):**
> "Now we'll run a live demo."

---

---

# DEMO SCRIPT (5 minutes)

## Before class — checklist (do these at home, verify everything works):

```
□ docker compose up  →  container starts, no errors
□ Open http://localhost:8000  →  page loads
□ Fill in normal scenario  →  prediction shows low probability
□ Fill in bad weather scenario  →  prediction shows higher probability
□ /health endpoint  →  {"status": "ok"}
□ Have terminal visible to show "docker compose up" output
```

---

## Demo flow (5 minutes in class):

### Step 1 — Show Docker running (1 min)
Open terminal, run:
```bash
docker compose up
```
Say: *"This is the entire system starting up in one command. The container loads the
pre-trained MLP model automatically."*

Show the startup log lines, wait for "Application startup complete."

---

### Step 2 — Open the web interface (30 sec)
Open browser → `http://localhost:8000`

Say: *"This HTML interface runs inside the same Docker container.
Let's try a normal landing first."*

---

### Step 3 — Normal landing scenario (1 min)
Fill in:
- Airport: `EDDF`, Runway: `25L`, WTC: `H`, Type: `A320`
- Glide slope: `3.0`, Runway length: `4000`
- Wind: `8 kt`, Visibility: `9999 m`, Temp: `18 °C`
- Month: `6`, Hour: `10`

Click **Predict**

Expected: low go-around probability (~15–25%)

Say: *"Frankfurt, clear weather, standard conditions — the model assigns low go-around risk."*

---

### Step 4 — High-risk scenario (1 min)
Fill in (or click **Fill Sample Input**, then modify):
- Airport: `KSAN`, Runway: `27`, WTC: `H`, Type: `B738`
- Glide slope: `3.0`, Runway length: `2865` (short runway!)
- Wind: `25 kt`, Wind direction: `090` (crosswind!), Gust: `35`
- Visibility: `800 m`, Precipitation: `RA`, Obscuration: `FG`

Click **Predict**

Expected: higher go-around probability

Say: *"San Diego's short runway, strong crosswind, low visibility and fog — the model
assigns higher risk. The probability bar and label update in real time."*

---

### Step 5 — Show API directly (30 sec)
Open new browser tab or show in terminal:
```bash
curl http://localhost:8000/health
```
Say: *"The system also exposes a JSON API endpoint that can be queried programmatically."*

---

### Step 6 — Wrap up (30 sec)
Say: *"To summarize: a single Docker container, a trained MLP model, a FastAPI backend,
and an HTML interface — all running with one command. Thank you."*
