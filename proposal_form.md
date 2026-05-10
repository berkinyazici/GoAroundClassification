# Project Proposal Form

## Go-Around Classification Using ADS-B and METAR Data

## 1. Team Members and Task Distribution

| Member | Planned Responsibilities |
|---|---|
| Furkan Güney<br>Student ID: 704241023 | Literature review, project framing, backend API, Docker packaging, HTML client interface, report writing, and presentation preparation. |
| Alper Berkin Yazıcı<br>Student ID: 704241020 | Data preprocessing, ADS-B and METAR feature engineering, model implementation/training, hyperparameter tuning, evaluation, and error analysis. |
| Joint Work | Final problem formulation, model selection, interpretation of results, live demo, and final presentation. |

## 2. Problem Definition

A go-around (missed approach) is an aborted landing during final approach or immediately after touchdown. Although it is a safety-preserving maneuver, it increases controller and pilot workload, reduces runway efficiency, and may propagate delays. Predicting go-arounds before touchdown is therefore an important aviation pattern recognition problem and is closely related to data-driven operational decision support in avionics and air traffic operations.

We formulate the task as binary classification. For each arrival, let

$$
y_i \in \{0, 1\},
$$

where $y_i = 1$ denotes a go-around and $y_i = 0$ denotes a successful landing. Each flight is represented by a feature vector

$$
x_i \in \mathbb{R}^{d}
$$

constructed from ADS-B trajectory information and METAR weather observations during final approach. The goal is to learn a decision function

$$
f(x_i) \rightarrow y_i.
$$

This study aims to investigate whether public ADS-B and METAR data can be used to predict go-around risk with satisfactory classification performance. The proposed study is aligned with the course scope, as it involves supervised learning, classification, probabilistic decision making, and empirical evaluation.

## 3. Dataset Description

The primary data source will be the publicly available *Large Landing Trajectory Dataset for Go-Around Analysis*, derived from OpenSky Network observations. We plan to use the `go_arounds_augmented.csv.gz` file because it already includes landing labels together with airport, runway, aircraft, and METAR-related attributes. The full dataset contains almost 9 million landings, more than 33,000 go-arounds, 176 airports, and 44 countries for 2019, which makes it suitable for both realistic modeling and controlled subsetting.

Planned feature groups are:

- **ADS-B/trajectory features:** altitude, vertical rate, ground speed, heading/track change, distance to runway threshold, runway alignment, and rolling statistics from the final approach segment.
- **Operational/context features:** airport/runway, aircraft type category, wake turbulence category, runway geometry, traffic density proxy, and time-of-day.
- **METAR features:** wind speed/direction, gust, visibility, pressure, temperature, and present weather indicators/codes.

Because go-arounds are rare, the dataset is imbalanced. We will therefore use class-aware training and evaluation metrics beyond accuracy. To keep the project manageable, the initial experiments will focus on a selected set of airports with sufficient go-around samples; extension to a broader multi-airport setting will be considered subsequently. Potential training and calibration trajectories will also be filtered, where necessary, using dataset metadata such as the number of approaches.

## 4. Proposed Methodology

**Preprocessing.** We will clean incomplete records, align variables to a consistent observation window, encode categorical variables, and standardize numerical features when needed. Data splits will be designed to avoid leakage across highly similar samples; we will prefer airport-aware and/or time-aware splitting strategies depending on the final subset.

**Models.** To align the project with core course topics, we plan to compare at least three classifiers:

1. Linear Discriminant Analysis (LDA) as a classical pattern recognition baseline,
2. one tree-based model (Decision Tree / Random Forest / Gradient Boosting),
3. a Multi-Layer Perceptron (MLP) as a neural baseline.

In addition, Logistic Regression will be included as an auxiliary probabilistic baseline.

For a linear probabilistic baseline,

$$
p(y_i = 1 \mid x_i) = \sigma(w^T x_i + b),
$$

where

$$
\sigma(z) = \frac{1}{1 + e^{-z}}.
$$

The predicted label is

$$
\hat{y}_i =
\begin{cases}
1, & p(y_i = 1 \mid x_i) \geq \tau, \\
0, & \text{otherwise.}
\end{cases}
$$

Training can be formulated with binary cross-entropy:

$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right].
$$

**Imbalance handling and ablation.** Since go-arounds are rare, class weighting and/or resampling strategies will be investigated. To justify the project title and quantify the value of weather information, an ablation study is planned to compare **ADS-B only** features against **ADS-B + METAR** features.

### System and Deployment Plan

To satisfy the course deployment requirements, the final trained classifier will be packaged in a single Dockerized application. A lightweight backend API, such as FastAPI, will expose a prediction endpoint that accepts either manual feature input or a small file-based request derived from the selected feature schema. A simple HTML-based web interface within the same container will allow the user to submit input features and receive the predicted class together with a confidence or probability score. This design is intended to ensure reproducibility, support in-class demonstration, and remain fully aligned with the course requirement of serving the model through an API and a minimal web front-end.

## 5. Literature Review

Prior work shows that go-arounds can be detected and analyzed from open ADS-B data. Proud presented an ADS-B-based go-around detection method and demonstrated that large-scale detection is feasible from crowd-sourced trajectory data. Figuet et al. showed that open ADS-B and meteorological data can be used for go-around prediction, while Monstein et al. later introduced a much larger public landing dataset that supports broader multi-airport experiments.

Kumar et al. studied go-arounds as a classification problem in commercial aviation using ADS-B data. Dhief et al. proposed a machine-learned go-around probability model for final approach and emphasized the difficulty of prediction under class imbalance and limited separability. More recently, Liu et al. studied real-time go-around prediction at JFK with sequence modeling and an operational web-based demonstration.

The proposed project builds on this line of work by using a public large-scale dataset with METAR attributes, by explicitly evaluating the contribution of weather features through an ADS-B-only vs. ADS-B+METAR comparison, and by comparing classical, tree-based, and neural classifiers within a reproducible deployment-oriented pipeline.

## 6. Evaluation Plan

We will evaluate the problem as binary classification. Since go-arounds are rare, accuracy alone is insufficient. Planned metrics are:

- Accuracy,
- Precision,
- Recall,
- F1-score,
- ROC-AUC,
- PR-AUC,
- confusion matrix.

We will use train/validation/test splits or cross-validation depending on the final subset size, with stratification when appropriate. Hyperparameters will be selected only on the validation data. In addition to comparing models, we will compare ADS-B-only and ADS-B+METAR feature sets, inspect false positives and false negatives, and analyze feature importance or airport/weather-specific error patterns as a secondary analysis.

## 7. Timeline

| Period | Planned Work |
|---|---|
| 08-15 Mar 2026 | Finalize proposal, confirm data source/file choice, set project scope, and prepare repository structure. |
| 16-29 Mar 2026 | Preprocess the augmented dataset, inspect class balance, define the feature schema, and prepare train/validation/test splits. |
| 30 Mar-12 Apr 2026 | Implement LDA and tree-based baselines; run initial experiments. |
| 13-26 Apr 2026 | Implement MLP baseline, perform tuning, and run ADS-B-only vs. ADS-B+METAR ablation experiments. |
| 27 Apr-03 May 2026 | Build backend API, Dockerize the project, and connect the HTML interface. |
| 04-10 May 2026 | Finalize report, prepare slides, rehearse live demo, and perform reproducibility checks. |

## References

[1] S. R. Proud, “Go-Around Detection Using Crowd-Sourced ADS-B Position Data,” *Aerospace*, vol. 7, no. 2, p. 16, 2020. doi: 10.3390/aerospace7020016.

[2] B. Figuet, R. Monstein, M. Waltert, and S. Barry, “Predicting Airplane Go-Arounds Using Machine Learning and Open-Source Data,” *Proceedings*, vol. 59, no. 1, p. 6, 2020. doi: 10.3390/proceedings2020059006.

[3] R. Monstein, B. Figuet, T. Krauth, M. Waltert, and M. Dettling, “Large Landing Trajectory Dataset for Go-Around Analysis,” *Engineering Proceedings*, vol. 28, no. 1, p. 2, 2022. doi: 10.3390/engproc2022028002.

[4] S. G. Kumar, S. J. Corrado, T. G. Puranik, and D. N. Mavris, “Classification and Analysis of Go-Arounds in Commercial Aviation Using ADS-B Data,” *Aerospace*, vol. 8, no. 10, p. 291, 2021. doi: 10.3390/aerospace8100291.

[5] I. Dhief, S. Alam, N. Lilith, and C. C. Mean, “A Machine Learned Go-Around Prediction Model Using Pilot-in-the-Loop Simulations,” *Transportation Research Part C: Emerging Technologies*, vol. 140, art. 103704, 2022. doi: 10.1016/j.trc.2022.103704.

[6] K. Liu, K. Ding, L. Dai, M. Hansen, K. Chan, and J. Schade, “Real-Time Go-Around Prediction: A Case Study of JFK Airport,” *arXiv preprint* arXiv:2405.12244, 2024. doi: 10.48550/arXiv.2405.12244.
