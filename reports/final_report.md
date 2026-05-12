# Go-Around Classification Using ADS-B and METAR Data

**BBL514E — Pattern Recognition Term Project**

**Furkan Güney** (704241023) · **Alper Berkin Yazıcı** (704241020)

Istanbul Technical University, Department of Computer Engineering

---

## Abstract

This project studies binary go-around risk classification using public ADS-B-derived landing records and METAR weather observations. A go-around is a rare but operationally important missed approach event. The source dataset contains approximately 9 million landings from 2019, but the current reproducible experiment uses a temporally split working subset with 283,355 training rows, 82,484 validation rows, and 82,474 test rows. The positive class rate is approximately 0.34-0.35 %, so the task is a severe rare-event detection problem rather than a conventional balanced classification problem.

We compare LDA, Logistic Regression, Random Forest, MLP, and LightGBM across three feature sets: context-only, context+METAR, and context+METAR with engineered features. Leakage-prone post-hoc fields such as `n_approaches` and `n_rwy_approached` are excluded from model inputs. To address imbalance, all positives are retained and negative training examples are downsampled to a 10:1 negative-to-positive ratio; applicable models also use class weighting or `scale_pos_weight`. Thresholds are tuned on validation using F2-score to favor recall. The final selected model is MLP with context+METAR features, achieving test ROC-AUC 0.5831 and test PR-AUC 0.0058. The system is deployed with FastAPI and an HTML interface that ranks input features by permutation importance, displays read-only derived features, and provides local what-if sensitivity analysis.

---

## 1. Introduction

A go-around is a safety-preserving maneuver where the flight crew aborts a landing attempt and climbs away for another approach. Although safe and often mandatory, it increases workload for pilots and controllers, affects runway throughput, and can propagate delays. Since go-arounds are rare, predictive modeling is difficult: normal landings and go-arounds share many similar observable conditions.

This project formulates go-around prediction as supervised binary classification:

```text
y = 1  go-around
y = 0  normal landing
```

The main research question is whether public ADS-B-derived operational features and METAR weather observations provide enough signal to rank landing attempts by go-around risk.

---

## 2. Dataset

The project uses the Large Landing Trajectory Dataset for Go-Around Analysis. The main input file is `go_arounds_augmented.csv.gz`, converted into Parquet during preprocessing. The source table contains landing-level records derived from ADS-B observations and enriched with METAR weather information.

Important available fields include:

- ADS-B-derived/context fields: `time`, `airport`, `runway`, `typecode`, `icaoaircrafttype`, `wtc`, `glide_slope_angle`, `rwy_length`
- METAR fields: `wind_speed_knts`, `wind_dir_deg`, `wind_gust_knts`, `visibility_m`, `temperature_deg`, `press_sea_level_p`, `press_p`, weather code columns
- Label: `has_ga` / `target`

The current processed temporal split is:

| Split | Rows | Go-arounds | Positive rate |
|---|---:|---:|---:|
| Train | 283,355 | 993 | 0.350 % |
| Validation | 82,484 | 280 | 0.339 % |
| Test | 82,474 | 283 | 0.343 % |

This class distribution makes accuracy misleading. A trivial normal-landing classifier would already exceed 99 % accuracy, so PR-AUC, recall, precision, F1/F2, and confusion matrix analysis are more informative.

---

## 3. Leakage Control

Several fields were excluded because they are identifiers, post-hoc information, or can lead to memorization:

```text
has_ga
target
n_approaches
n_rwy_approached
icao24
callsign
registration
raw time
```

The most important exclusion is `n_approaches`: a go-around naturally increases the number of approaches, so using this field would leak the answer after the event. This was one reason earlier experiments produced unrealistically perfect metrics for some models.

---

## 4. Feature Sets

### 4.1 Context Only

Numeric:

```text
glide_slope_angle
rwy_length
month
day_of_week
hour_utc
```

Categorical:

```text
airport
runway
typecode
icaoaircrafttype
wtc
has_intersection
airport_country
airport_region
operator_country
operator_region
```

### 4.2 Context + METAR

Adds weather measurements and weather codes:

```text
wind_speed_knts
wind_dir_deg
wind_gust_knts
visibility_m
temperature_deg
press_sea_level_p
press_p
weather_intensity
weather_precipitation
weather_desc
weather_obscuration
weather_other
```

### 4.3 Context + METAR + Engineered

The engineered feature set was added after observing that raw tabular features had limited minority-class separability.

Runway-relative wind features:

| Feature | Derived from |
|---|---|
| `runway_heading_deg` | `runway` |
| `headwind_knts` | `runway`, `wind_dir_deg`, `wind_speed_knts` |
| `tailwind_knts` | `runway`, `wind_dir_deg`, `wind_speed_knts` |
| `crosswind_knts` | `runway`, `wind_dir_deg`, `wind_speed_knts` |

Weather severity features:

| Feature | Derived from |
|---|---|
| `gust_spread_knts` | `wind_gust_knts - wind_speed_knts` |
| `low_visibility_flag` | `visibility_m < 5000` |
| `very_low_visibility_flag` | `visibility_m < 1500` |
| `strong_wind_flag` | `wind_speed_knts >= 25` |
| `strong_crosswind_flag` | `crosswind_knts >= 15` |
| `tailwind_flag` | `tailwind_knts >= 5` |
| `adverse_weather_flag` | weather codes, low visibility, strong wind/crosswind |

Train-only risk encodings:

| Feature | Derived from |
|---|---|
| `airport_risk_train` | train-set go-around rate by airport |
| `runway_risk_train` | train-set go-around rate by runway |
| `typecode_risk_train` | train-set go-around rate by aircraft type |

These risk encodings are calculated only from the training split and then applied to validation/test. This avoids using validation/test labels during feature creation.

---

## 5. Models and Training

The following classifiers were evaluated:

- Linear Discriminant Analysis
- Logistic Regression
- Random Forest
- Multi-Layer Perceptron
- LightGBM

Training choices:

- All positive training samples are kept.
- Negative samples are downsampled to a 10:1 negative-to-positive ratio.
- Logistic Regression uses `class_weight="balanced"`.
- Random Forest uses `class_weight="balanced_subsample"`.
- LightGBM uses `scale_pos_weight`.
- Threshold tuning uses F2-score on validation, giving recall more weight than precision.
- Model selection is still based primarily on validation PR-AUC because PR-AUC is the most suitable ranking metric for rare-event classification.

---

## 6. Results

The latest model comparison is summarized below. Models are ordered by validation PR-AUC.

| Model | Feature Set | Val ROC-AUC | Val PR-AUC | Test ROC-AUC | Test PR-AUC | Test F1 | Test Precision | Test Recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **MLP** | **context_metar** | **0.6068** | **0.0098** | 0.5831 | 0.0058 | 0.0159 | 0.0091 | 0.0636 |
| LightGBM | context_metar_engineered | 0.6189 | 0.0093 | 0.6037 | 0.0056 | 0.0166 | 0.0098 | 0.0565 |
| LDA | context_metar | 0.6055 | 0.0082 | 0.6238 | 0.0064 | 0.0164 | 0.0098 | 0.0495 |
| LDA | context_metar_engineered | 0.6235 | 0.0081 | **0.6515** | **0.0067** | 0.0181 | 0.0100 | **0.0954** |
| Logistic Regression | context_metar | 0.6123 | 0.0081 | 0.6197 | 0.0066 | **0.0202** | **0.0114** | 0.0848 |
| Random Forest | context_metar_engineered | **0.6309** | 0.0078 | 0.6357 | 0.0063 | 0.0153 | 0.0087 | 0.0601 |
| Logistic Regression | context_metar_engineered | 0.6233 | 0.0076 | 0.6463 | 0.0066 | 0.0179 | 0.0102 | 0.0742 |
| MLP | context_metar_engineered | 0.6225 | 0.0072 | 0.6327 | 0.0063 | 0.0146 | 0.0088 | 0.0424 |

Key observations:

1. Engineered features improved ROC-AUC for several models, especially LDA and Logistic Regression.
2. The best validation PR-AUC still came from MLP with the non-engineered context+METAR feature set.
3. Absolute PR-AUC remains low because the positive class rate is only about 0.34 %.
4. F2 thresholding increases recall pressure but produces many false positives, keeping precision low.

---

## 7. Final Model

The selected final model is:

```text
Model: MLP
Feature set: context_metar
Threshold: 0.2116
```

Final test metrics:

| Metric | Value |
|---|---:|
| Accuracy | 0.9729 |
| Precision | 0.0091 |
| Recall | 0.0636 |
| F1 | 0.0159 |
| ROC-AUC | 0.5831 |
| PR-AUC | 0.0058 |

Confusion matrix:

|  | Predicted Normal | Predicted Go-Around |
|---|---:|---:|
| Actual Normal | 80,224 | 1,967 |
| Actual Go-Around | 265 | 18 |

Interpretation:

- The model catches 18 of 283 go-arounds in the test set.
- Recall is 6.36 %, higher than earlier strict-threshold runs but still low.
- Precision is 0.91 %, meaning most predicted go-around alerts are false positives.
- This is expected in rare-event detection when available landing-level features only weakly separate the classes.

---

## 8. Feature Importance and Interface Explainability

Because the final model is an MLP, there is no native tree-style feature importance. We therefore computed model-agnostic permutation importance on a test sample using average precision as the scoring metric. The top-ranked features were:

| Rank | Feature | Relative importance |
|---:|---|---:|
| 1 | `wind_gust_knts` | 1.000 |
| 2 | `weather_desc` | 0.845 |
| 3 | `weather_other` | 0.625 |
| 4 | `weather_precipitation` | 0.555 |
| 5 | `operator_region` | 0.541 |
| 6 | `wind_speed_knts` | 0.392 |
| 7 | `weather_intensity` | 0.193 |
| 8 | `icaoaircrafttype` | 0.165 |

The web interface now uses this information directly:

- Input fields are sorted by final-model permutation importance.
- Each input displays its rank and relative impact.
- Read-only derived features are shown with dependency labels.
- Changing raw fields updates backend-computed derived values.
- After prediction, a local what-if sensitivity panel shows how increasing/decreasing numeric fields changes the go-around probability for the current input.

This makes the demo more interpretable: users can see both global model importance and local input sensitivity.

---

## 9. Deployment

The application is deployed as a FastAPI service with a simple HTML frontend.

Endpoints:

| Endpoint | Purpose |
|---|---|
| `/` | Web interface |
| `/health` | Health check |
| `/predict` | Single-flight prediction |
| `/derived-features` | Backend-computed read-only feature values |
| `/feature-importance` | Final-model permutation importance |
| `/sensitivity` | Local what-if probability sensitivity |

The frontend allows users to enter operational and METAR fields, inspect derived features, run prediction, and interpret the result through feature importance and sensitivity panels.

---

## 10. Discussion

The results show that this dataset can provide weak but non-random go-around risk ranking. However, the task remains very hard. Several reasons explain the low precision and recall:

1. **Extreme class imbalance:** only about 0.34 % of rows are positive.
2. **Weak separability:** many go-arounds occur under conditions also seen in normal landings.
3. **Landing-level aggregation:** the current table lacks per-second trajectory dynamics such as vertical rate stability, speed profile, runway alignment over time, or altitude deviation.
4. **Operational decisions are complex:** go-arounds depend on pilot/controller decisions, runway occupancy, traffic sequencing, and aircraft state variables not fully represented in the tabular data.

The engineered features help reveal additional signal, especially in ROC-AUC, but they do not fully solve the minority-class detection problem.

---

## 11. Conclusion

This project built an end-to-end go-around classification pipeline using ADS-B-derived tabular data and METAR weather observations. Leakage-prone fields were removed, imbalance-aware training was added, engineered weather/runway/risk features were created, and an explainable web interface was deployed.

The best model remains modest in absolute performance, with PR-AUC 0.0058 and recall 0.0636 on the test set. This is still above the no-skill baseline but not suitable for operational aviation use. The main scientific conclusion is that landing-level ADS-B-derived context and METAR weather provide limited but measurable signal; stronger results likely require extracting time-series features from raw ADS-B approach trajectories.

Future work should focus on:

- per-second ADS-B trajectory feature extraction,
- runway-relative approach stability metrics,
- airport-specific calibration,
- alternative threshold policies for operational objectives,
- sequence models such as LSTM/Transformer architectures.

---

## References

[1] S. R. Proud, "Go-Around Detection Using Crowd-Sourced ADS-B Position Data," *Aerospace*, vol. 7, no. 2, p. 16, 2020.

[2] B. Figuet, R. Monstein, M. Waltert, and S. Barry, "Predicting Airplane Go-Arounds Using Machine Learning and Open-Source Data," *Proceedings*, vol. 59, no. 1, p. 6, 2020.

[3] R. Monstein, B. Figuet, T. Krauth, M. Waltert, and M. Dettling, "Large Landing Trajectory Dataset for Go-Around Analysis," *Engineering Proceedings*, vol. 28, no. 1, p. 2, 2022.

[4] S. G. Kumar, S. J. Corrado, T. G. Puranik, and D. N. Mavris, "Classification and Analysis of Go-Arounds in Commercial Aviation Using ADS-B Data," *Aerospace*, vol. 8, no. 10, p. 291, 2021.

[5] H. He and E. A. Garcia, "Learning from Imbalanced Data," *IEEE Transactions on Knowledge and Data Engineering*, vol. 21, no. 9, pp. 1263-1284, 2009.

[6] J. Davis and M. Goadrich, "The Relationship Between Precision-Recall and ROC Curves," in *Proceedings of ICML*, 2006.
