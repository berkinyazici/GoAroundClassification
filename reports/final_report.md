# Go-Around Classification Using ADS-B and METAR Data

**Team members:** Furkan Güney (704241023), Alper Berkin Yazıcı (704241020)

## Abstract

This study addressed the prediction of aircraft go-arounds as a binary pattern-recognition problem. Each landing record was represented by ADS-B-derived operational attributes, aircraft and runway context, and METAR weather variables from a public augmented landing dataset. Classical and machine-learning classifiers were implemented, including Logistic Regression, Linear Discriminant Analysis, tree-based models, a Multi-Layer Perceptron, and LightGBM. The pipeline cleaned the augmented table, removed likely training/calibration flights with more than two approaches, derived calendar features from timestamps, encoded categorical attributes, and handled severe class imbalance with class weighting or positive-class scaling. Models were compared using accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, and confusion matrices, with threshold selection based on validation F1-score. An ablation study compared ADS-B-only features with ADS-B plus METAR features to quantify the value of weather information. The project also delivered a Dockerized FastAPI service and a browser-based client that accepts landing features and returns a go-around probability. The resulting system provides a reproducible academic experiment and a live demonstration platform, while remaining a decision-support prototype rather than an operational safety tool.

## 1. Introduction and Literature Review

A go-around is an aborted landing or missed approach. Although it is a safe and standard maneuver, it increases flight-crew workload, controller workload, fuel burn, runway occupancy uncertainty, and delay propagation. Predicting go-around likelihood before touchdown is therefore an important aviation pattern-recognition task.

This project formulates go-around prediction as supervised binary classification. It focuses on a realistic term-project scope: tabular records derived from open ADS-B landing data and enriched with METAR weather, airport, runway, aircraft, and operator attributes. Raw trajectory reconstruction and sequence modeling are left as future work.

Related work has shown that open ADS-B data can be used to detect and analyze go-arounds at scale. Proud (2020) introduced a crowd-sourced ADS-B go-around detection method. Figuet et al. (2020) studied machine-learning prediction from open-source data. Monstein et al. (2022) released a larger landing-trajectory dataset that enables multi-airport analysis. Kumar et al. (2021) treated go-around identification and analysis as a classification problem. Dhief et al. (2022) studied machine-learned go-around probability under class imbalance. Recent work by Liu et al. (2024) explored real-time prediction and web-based demonstration for JFK. Broader pattern-recognition and imbalanced-learning foundations are covered by Bishop (2006), Duda et al. (2001), Fawcett (2006), and Saito and Rehmsmeier (2015).

The contribution of this project is a reproducible, deployment-oriented implementation that (i) compares multiple classifiers, (ii) evaluates imbalanced binary metrics, (iii) explicitly tests ADS-B-only vs ADS-B+METAR features, and (iv) serves the selected model through a Dockerized API and HTML interface.

## 2. Materials and Methods

### 2.1 Dataset

The project uses the public go-around dataset from Zenodo record 7148117. The required files are `go_arounds_augmented.csv.gz`, `go_arounds_agg.csv.gz`, and `validation_table.xlsx`. The augmented table contains the binary go-around label (`has_ga`) plus context fields such as time, airport, runway, aircraft type, runway geometry, and METAR-derived weather variables.

The canonical target used by the implementation is:

\[
y_i \in \{0,1\}, \quad y_i=1 \text{ for go-around}, \quad y_i=0 \text{ for normal landing}.
\]

### 2.2 Preprocessing

The pipeline performs the following steps:

1. load the augmented CSV.GZ or parquet file;
2. convert `has_ga` into the canonical integer `target` column;
3. drop records with missing target labels;
4. remove likely training/calibration flights where `n_approaches > 2`;
5. parse `time` as UTC datetime;
6. derive `month`, `day_of_week`, and `hour_utc`;
7. coerce numeric fields to numeric dtype;
8. standardize missing categorical and list-like weather fields to safe strings.

### 2.3 Feature sets

The ADS-B/context feature set includes approach counts, runway geometry, aircraft and operator categories, airport/runway identifiers, and time features. The ADS-B+METAR feature set adds wind speed, wind direction, gust, visibility, temperature, pressure, and weather-code categories. This enables an ablation study that directly answers whether weather information improves go-around prediction.

### 2.4 Mathematical formulation

Each landing is represented by a feature vector \(x_i \in \mathbb{R}^d\) after preprocessing and categorical encoding. A probabilistic classifier estimates:

\[
\hat{p}_i = P(y_i=1 \mid x_i).
\]

The decision rule is:

\[
\hat{y}_i =
\begin{cases}
1, & \hat{p}_i \geq \tau,\\
0, & \hat{p}_i < \tau,
\end{cases}
\]

where \(\tau\) is selected on the validation set. For Logistic Regression:

\[
P(y_i=1\mid x_i)=\sigma(w^T x_i+b), \quad \sigma(z)=\frac{1}{1+e^{-z}}.
\]

The binary cross-entropy objective is:

\[
\mathcal{L}=-\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log(\hat{p}_i)+(1-y_i)\log(1-\hat{p}_i)\right].
\]

Class imbalance is handled with class weights or positive-class scaling.

## 3. Experimental Setup

The preferred split is time-based: training before 2019-09-01, validation from 2019-09-01 to 2019-10-31, and testing from 2019-11-01 onward. If a small sample does not contain all time windows or both classes, the script falls back to stratified random splitting for development only.

Implemented models are Logistic Regression, Linear Discriminant Analysis, Decision Tree, Random Forest-style tree ensemble support, MLP, and LightGBM. The main metrics are accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, and confusion matrix. PR-AUC and recall are emphasized because go-arounds are rare.

The reproducible commands are:

```bash
python -m src.cli make-interim
python -m src.cli make-splits
python scripts/run_ablation.py --models logreg tree lightgbm
python -m src.cli evaluate
```

## 4. Results

The repository writes final metrics to `reports/metrics/evaluation_metrics.json` and figures to `reports/figures/`. For the submitted PDF, insert the full-data numerical results after executing the final workflow. The expected result tables are:

| Model | Feature set | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | ADS-B only | fill after run | fill after run | fill after run | fill after run | fill after run | fill after run |
| Logistic Regression | ADS-B + METAR | fill after run | fill after run | fill after run | fill after run | fill after run | fill after run |
| Tree/LightGBM | ADS-B + METAR | fill after run | fill after run | fill after run | fill after run | fill after run | fill after run |

Include the generated confusion matrix, precision-recall curve, and ROC curve in the final PDF.

## 5. Deployment

The system is deployed with FastAPI and a simple HTML interface. The `/predict` endpoint accepts a JSON feature dictionary and returns the predicted class, human-readable label, go-around probability, threshold, and model name. Docker runs the backend and client in the same container.

## 6. Conclusion

The project implements a complete pattern-recognition workflow for go-around classification using ADS-B and METAR tabular data. It provides reproducible preprocessing, several classifiers, imbalanced evaluation metrics, ADS-B vs METAR ablation, and a Dockerized live demo. Limitations include the absence of raw trajectory sequence features and the fact that this system is a prototype for academic analysis, not an operational aviation safety system. Future work should include temporal trajectory models, airport-held-out validation, calibration analysis, and operational human-factors evaluation.

## References

1. S. R. Proud, “Go-Around Detection Using Crowd-Sourced ADS-B Position Data,” *Aerospace*, 7(2), 16, 2020.
2. B. Figuet, R. Monstein, M. Waltert, and S. Barry, “Predicting Airplane Go-Arounds Using Machine Learning and Open-Source Data,” *Proceedings*, 59(1), 6, 2020.
3. R. Monstein, B. Figuet, T. Krauth, M. Waltert, and M. Dettling, “Large Landing Trajectory Dataset for Go-Around Analysis,” *Engineering Proceedings*, 28(1), 2, 2022.
4. S. G. Kumar, S. J. Corrado, T. G. Puranik, and D. N. Mavris, “Classification and Analysis of Go-Arounds in Commercial Aviation Using ADS-B Data,” *Aerospace*, 8(10), 291, 2021.
5. I. Dhief, S. Alam, N. Lilith, and C. C. Mean, “A Machine Learned Go-Around Prediction Model Using Pilot-in-the-Loop Simulations,” *Transportation Research Part C*, 140, 103704, 2022.
6. K. Liu, K. Ding, L. Dai, M. Hansen, K. Chan, and J. Schade, “Real-Time Go-Around Prediction: A Case Study of JFK Airport,” arXiv:2405.12244, 2024.
7. C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006.
8. R. O. Duda, P. E. Hart, and D. G. Stork, *Pattern Classification*, Wiley, 2001.
9. T. Fawcett, “An Introduction to ROC Analysis,” *Pattern Recognition Letters*, 27(8), 861–874, 2006.
10. T. Saito and M. Rehmsmeier, “The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets,” *PLOS ONE*, 10(3), e0118432, 2015.
