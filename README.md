# ICC T20 World Cup 2026 Prediction

A machine learning project that predicts match outcomes and the tournament winner for the ICC T20 World Cup 2026 using historical T20 International match data.

## Project Overview

This project is an end to end machine learning system for predicting match outcomes in the ICC 2026 T20 World Cup using historical T20I data. It covers the full pipeline from raw data cleaning and preprocessing to feature engineering, model training, evaluation, and tournament level prediction. Four different machine learning models **[Logistic Regression, Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors (KNN)]** are trained and compared to study performance trade offs, generalisation behaviour, and overfitting, rather than relying on a single algorithm.

The system includes detailed evaluation using accuracy metrics, cross validation, confusion matrices, and clear visualisations to compare model behaviour. Results show how simpler linear models and more complex ensemble models perform under the same data constraints, highlighting stability versus capacity. Predictions can be generated for individual matches as well as full tournament simulations, with outputs exported for further analysis and inspection.

The project is built with a clean, reproducible structure, separating data preprocessing, training, evaluation, visualisation, and prediction logic. It is designed to be extensible, allowing new features, models, or updated datasets to be added easily as new matches are played.

### Key Highlights

- T20I data spanning **2005–2024**
- **812 training matches**, **204 test matches**
- Binary classification: **Team 1 Win vs Team 2 Win**
- Models evaluated:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
- Best performing model: **Logistic Regression**

---

## Predicted Winner: **INDIA**

Based on the Logistic Regression model, which demonstrated the best generalization performance:

- **Test Accuracy**: **61.76%**
- **Cross-Validation Mean Accuracy**: **69.09%**

India is predicted as the most likely winner of the ICC T20 World Cup 2026 based on historical trends and engineered team-level features.

---

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Run the complete pipeline**

```bash
# Step 1: Preprocess data (creates all datasets)
python data_preprocessing.py

# Step 2: Train models
python train_logistic_regression.py
python train_random_forest.py

# Step 3: Compare all models
python compare_models.py

# Step 4: Generate World Cup predictions
python predict_world_cup.py
```

## What Each Script Does

### 1. `data_preprocessing.py`

- Processes raw T20I match data (2005-2024)
- Engineer team-level features
- Generates ICC team rankings
- Filters data from 2010 onwards for better relevance

**Outputs:**

- `datasets/results.csv`
- `datasets/training_data.csv`
- `datasets/icc_rankings.csv`

### 2. `logistic_regression.py`

- Trains a Logistic Regression classifier
- Evaluates model performance on test data
- Uses standardized features
- Saves trained model for predictions

**Outputs:**

- `models/random_forest_model.pkl` - Trained model
- `models/scaler.pkl` - Feature scaler
- `visualizations/random_forest_analysis.png` - Performance plots

### 2. `train_random_forest.py`

- Trains Random Forest classifier
- Evaluates performance
- Demonstrates overfitting behavior
- Saves trained model and feature scalar

**Outputs:**

- `models/random_forest_model.pkl` - Trained model
- `models/scaler.pkl` - Feature scaler
- `visualizations/random_forest_analysis.png` - Performance plots

### 3. `compare_models.py`

- Compares four machine learning models:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)

- Evaluates models using:
  - Train accuracy
  - Test accuracy
  - Cross-validation mean and standard deviation
  - Precision, Recall, and F1-score

- Identifies the best-performing and best-generalizing model

**Outputs:**

- `datasets/model_comparison_results.csv`
- `datasets/detailed_model_metrics.csv`
- `visualizations/model_comparison.png`
- `visualizations/confusion_matrices.png`

### 4. `predict_world_cup.py`

- Predicts all group stage matches
- Simulates knockout rounds
- Predicts tournament winner
- Shows detailed match analysis
- Displays match-level prediction confidence

**Outputs:**

- `datasets/group_stage_predictions.csv` - All predictions
- Console output with detailed predictions

---

## Project Structure After Running All Scripts

```
ICC-2026-WC-prediction/
│
├── datasets/
│   ├── results.csv                         # Processed match results
│   ├── training_data.csv                   # Training features
│   ├── icc_rankings.csv                    # Team rankings
│   ├── group_stage_predictions.csv         # Tournament predictions
│   ├── model_comparison_results.csv        # Model metrics
│   └── detailed_model_metrics.csv          # Detailed metrics
│
├── models/
│   ├── logistic_regression_model.pkl       # Trained Logistic Regression model
│   ├── random_forest_model.pkl             # Trained Random Forest model
│   └── scaler.pkl                          # Feature scaler
│
├── visualizations/
│   ├── random_forest_analysis.png          # RF performance
│   ├── model_comparison.png                # Model comparison
│   └── confusion_matrices.png              # Confusion matrices
│
├── data_preprocessing.py                   # Data processing
├── train_logical_regression.py                  # Model training
├── train_random_forest.py                  # Model training
├── compare_models.py                       # Model comparison
├── predict_world_cup.py                    # Predictions
├── requirements.txt                        # Dependencies
├── README.md                               # Main documentation
```

---

## Data Sources

### Primary Dataset

- **All T20 Internationals Dataset (2005–2024)** by Bhuvanesh Prasad
- Kaggle: https://www.kaggle.com/datasets/bhuvaneshprasad/all-t20-internationals-dataset-2005-to-2023/data

Includes:

- Match dates and results
- Participating teams
- Runs, wickets, and outcomes

### Processed Datasets

- `training_data.csv` – Engineered features for model training
- `results.csv` – Filtered T20I matches
- `icc_rankings.csv` – Team ranking information
- `group_stage_predictions.csv` – Predicted group-stage outcomes

---

## Models and Performance

### Model Comparison Summary

| Model                   | Train Accuracy | Test Accuracy | CV Mean    | CV Std     |
| ----------------------- | -------------- | ------------- | ---------- | ---------- |
| Random Forest           | 96.43%         | 60.78%        | 64.66%     | ±2.78%     |
| **Logistic Regression** | **71.43%**     | **61.76%**    | **69.09%** | **±1.45%** |
| SVM                     | 72.41%         | 61.27%        | 65.89%     | ±1.80%     |
| KNN                     | 100.00%        | 60.29%        | 60.71%     | ±2.89%     |

### Detailed Metrics (Team 1 Wins)

| Model                   | Precision | Recall   | F1-Score |
| ----------------------- | --------- | -------- | -------- |
| Random Forest           | 0.61      | 0.63     | 0.62     |
| **Logistic Regression** | **0.61**  | **0.68** | **0.65** |
| SVM                     | 0.61      | 0.65     | 0.63     |
| KNN                     | 0.60      | 0.64     | 0.62     |

---

### Overfitting Analysis

- **Random Forest**: Very high training accuracy indicates overfitting
- **KNN**: Perfect training accuracy but weak generalization
- **Logistic Regression**: Best balance between bias and variance
- **SVM**: Stable but slightly weaker cross-validation score

---

## Tournament Prediction Strategy

- Group-stage matches predicted using pairwise team comparisons
- Knockout rounds simulated using model probabilities
- Confidence values derived from model output probabilities

---

## Visualizations

Automatically generated using `compare_models.py`:

- `model_comparison.png` – Accuracy and cross-validation comparison
- `confusion_matrices.png` – Confusion matrices for all models
- `random_forest_analysis.png` – Metrics of Random Forest model
- `logistic_regression_analysis.png` – Metrics of Logistic Regression model
  Saved under: visualizations/

---

## Example Output

```
C:\Users\Asus\Documents\GitHub\Cricket World Cup Predictor\ICC-2026-WC-prediction>python data_preprocessing.py
Loading matches data...
Total matches: 2592
Main teams matches: 1181

Results dataset shape: (1126, 6)
        date    Team_1  ...                                             Margin                     Ground
0 2008-02-01     India  ...  Australia won by 9 wickets (with 52 balls rema...       Melbourne, Australia
1 2008-02-07   England  ...                             England won by 50 runs  Christchurch, New Zealand
2 2008-08-04  Scotland  ...  Netherlands won by 5 wickets (with 12 balls re...           Belfast, Ireland
3 2008-10-10  Zimbabwe  ...  Sri Lanka won by 5 wickets (with 6 balls remai...          King City, Canada
4 2008-10-10  Pakistan  ...                            Pakistan won by 35 runs          King City, Canada

[5 rows x 6 columns]

Saved results to datasets/results.csv

Filtered results (2010+): 1016 matches
Creating training dataset...
Processed 200/1016 matches
Processed 300/1016 matches
Processed 400/1016 matches
Processed 500/1016 matches
Processed 700/1016 matches
Processed 800/1016 matches
Processed 900/1016 matches
Processed 1000/1016 matches
Processed 1100/1016 matches

Training dataset shape: (1016, 15)
        Team_1       Team_2       date  team1_matches  ...  h2h_total  h2h_team1_wins  h2h_win_rate  team1_wins_match
0      Ireland     Scotland 2010-02-11              4  ...          0               0           0.5                 1
1  Afghanistan      Ireland 2010-02-01              1  ...          0               0           0.5                 0
2       Canada      Ireland 2010-02-03              1  ...          0               0           0.5                 1
3   Bangladesh  New Zealand 2010-02-03              1  ...          0               0           0.5                 0
4       Canada  Afghanistan 2010-02-04              2  ...          0               0           0.5                 0

[5 rows x 15 columns]

Saved training data to datasets/training_data.csv

=== Current T20I Rankings (Based on Data) ===
    Position           Team  Matches  Win_Rate  Recent_Form      Rating
0          1          India      188  0.680851     0.750000  722.340426
1          2      Australia      157  0.560510     0.600000  584.203822
2          3     Bangladesh      141  0.390071     0.700000  576.028369
3          4    New Zealand      168  0.565476     0.550000  556.190476
4          5        England      152  0.559211     0.550000  553.684211
5          6        Namibia       33  0.545455     0.550000  548.181818
6          7   South Africa      143  0.552448     0.450000  490.979021
7          8    West Indies      162  0.438272     0.500000  475.308642
8          9    Afghanistan      106  0.603774     0.350000  451.509434
9         10      Sri Lanka      156  0.435897     0.450000  444.358974
10        11       Pakistan      194  0.577320     0.350000  440.927835
11        12          Nepal       27  0.407407     0.450000  432.962963
12        13    Netherlands       67  0.477612     0.350000  401.044776
13        14       Scotland       54  0.388889     0.350000  365.555556
14        15        Ireland      111  0.387387     0.350000  364.954955
15        16         Canada       15  0.333333     0.333333  333.333333
16        17       Zimbabwe      120  0.266667     0.250000  256.666667
17        18           Oman       36  0.250000     0.250000  250.000000
18        19  United States        0  0.000000     0.000000    0.000000
19        20          Italy        2  0.000000     0.000000    0.000000

Saved rankings to datasets/icc_rankings.csv

Data preprocessing complete!

C:\Users\Asus\Documents\GitHub\Cricket World Cup Predictor\ICC-2026-WC-prediction>python train_logistic_regression.py
============================================================
ICC 2026 WORLD CUP PREDICTION - LOGISTIC REGRESSION MODEL
============================================================

1. Loading training data...
Training data shape: (1016, 15)

Features:
['Team_1', 'Team_2', 'date', 'team1_matches', 'team1_wins', 'team1_win_rate', 'team1_recent_form', 'team2_matches', 'team2_wins', 'team2_win_rate', 'team2_recent_form', 'h2h_total', 'h2h_team1_wins', 'h2h_win_rate', 'team1_wins_match']

2. Preparing features and target...
Features shape: (1016, 11)
Target shape: (1016,)
Target distribution:
team1_wins_match
1    518
0    498
Name: count, dtype: int64

3. Splitting data (80-20 split)...
Training set: (812, 11)
Test set: (204, 11)

4. Scaling features...

5. Training Logistic Regression model...
Model trained successfully

6. Making predictions...

============================================================
MODEL PERFORMANCE
============================================================
Training Accuracy: 71.43%
Test Accuracy: 61.76%

============================================================
CLASSIFICATION REPORT (Test Set)
============================================================
              precision    recall  f1-score   support

 Team 2 Wins       0.62      0.55      0.59       100
 Team 1 Wins       0.61      0.68      0.65       104

    accuracy                           0.62       204
   macro avg       0.62      0.62      0.62       204
weighted avg       0.62      0.62      0.62       204


Confusion Matrix:
[[55 45]
 [33 71]]

============================================================
FEATURE COEFFICIENTS
============================================================
team1_recent_form        :  0.6818 (|0.6818|)
team2_recent_form        : -0.5902 (|0.5902|)
team2_matches            : -0.5888 (|0.5888|)
team2_win_rate           : -0.3657 (|0.3657|)
h2h_team1_wins           : -0.3545 (|0.3545|)
team1_wins               :  0.2737 (|0.2737|)
h2h_total                :  0.2515 (|0.2515|)
team1_matches            :  0.1776 (|0.1776|)
team1_win_rate           :  0.1430 (|0.1430|)
h2h_win_rate             : -0.1244 (|0.1244|)
team2_wins               :  0.1212 (|0.1212|)

7. Creating visualizations...
Saved visualization: visualizations/logistic_regression_analysis.png
Saved model: models/logistic_regression_model.pkl
Saved scaler: models/scaler_lr.pkl

============================================================
LOGISTIC REGRESSION MODEL TRAINING COMPLETE
============================================================

C:\Users\Asus\Documents\GitHub\Cricket World Cup Predictor\ICC-2026-WC-prediction>python train_random_forest.py
============================================================
ICC 2026 WORLD CUP PREDICTION - RANDOM FOREST MODEL
============================================================

1. Loading training data...
Training data shape: (1016, 15)

Features:
['Team_1', 'Team_2', 'date', 'team1_matches', 'team1_wins', 'team1_win_rate', 'team1_recent_form', 'team2_matches', 'team2_wins', 'team2_win_rate', 'team2_recent_form', 'h2h_total', 'h2h_team1_wins', 'h2h_win_rate', 'team1_wins_match']

2. Preparing features and target...
Features shape: (1016, 11)
Target shape: (1016,)
Target distribution:
team1_wins_match
1    518
0    498
Name: count, dtype: int64

3. Splitting data (80-20 split)...
Training set: (812, 11)
Test set: (204, 11)

4. Scaling features...

5. Training Random Forest model...
Model trained successfully

6. Making predictions...

============================================================
MODEL PERFORMANCE
============================================================
Training Accuracy: 96.43%
Test Accuracy: 60.78%

============================================================
CLASSIFICATION REPORT (Test Set)
============================================================
              precision    recall  f1-score   support

 Team 2 Wins       0.60      0.58      0.59       100
 Team 1 Wins       0.61      0.63      0.62       104

    accuracy                           0.61       204
   macro avg       0.61      0.61      0.61       204
weighted avg       0.61      0.61      0.61       204


Confusion Matrix:
[[58 42]
 [38 66]]

============================================================
FEATURE IMPORTANCE
============================================================
team1_win_rate           : 0.1367
team2_win_rate           : 0.1319
team2_recent_form        : 0.1290
team1_recent_form        : 0.1283
team2_matches            : 0.0833
team2_wins               : 0.0793
team1_wins               : 0.0778
team1_matches            : 0.0742
h2h_win_rate             : 0.0671
h2h_total                : 0.0568
h2h_team1_wins           : 0.0356

7. Creating visualizations...
Saved visualization: visualizations/random_forest_analysis.png
Saved model: models/random_forest_model.pkl
Saved scaler: models/scaler.pkl

============================================================
RANDOM FOREST MODEL TRAINING COMPLETE
============================================================

C:\Users\Asus\Documents\GitHub\Cricket World Cup Predictor\ICC-2026-WC-prediction>python compare_models.py
======================================================================
ICC 2026 WORLD CUP PREDICTION - MODEL COMPARISON
======================================================================

Loading data...
Training samples: 812
Test samples: 204

======================================================================
TRAINING AND EVALUATING MODELS
======================================================================

Random Forest:
----------------------------------------------------------------------
Training Accuracy: 0.9643
Test Accuracy: 0.6078
Cross-Val Mean: 0.6466 (+/- 0.0278)

Classification Report:
              precision    recall  f1-score   support

 Team 2 Wins       0.60      0.58      0.59       100
 Team 1 Wins       0.61      0.63      0.62       104

    accuracy                           0.61       204
   macro avg       0.61      0.61      0.61       204
weighted avg       0.61      0.61      0.61       204


Logistic Regression:
----------------------------------------------------------------------
Training Accuracy: 0.7143
Test Accuracy: 0.6176
Cross-Val Mean: 0.6909 (+/- 0.0145)

Classification Report:
              precision    recall  f1-score   support

 Team 2 Wins       0.62      0.55      0.59       100
 Team 1 Wins       0.61      0.68      0.65       104

    accuracy                           0.62       204
   macro avg       0.62      0.62      0.62       204
weighted avg       0.62      0.62      0.62       204


SVM:
----------------------------------------------------------------------
Training Accuracy: 0.7241
Test Accuracy: 0.6127
Cross-Val Mean: 0.6589 (+/- 0.0180)

Classification Report:
              precision    recall  f1-score   support

 Team 2 Wins       0.61      0.57      0.59       100
 Team 1 Wins       0.61      0.65      0.63       104

    accuracy                           0.61       204
   macro avg       0.61      0.61      0.61       204
weighted avg       0.61      0.61      0.61       204


K-Nearest Neighbors:
----------------------------------------------------------------------
Training Accuracy: 1.0000
Test Accuracy: 0.6029
Cross-Val Mean: 0.6071 (+/- 0.0289)

Classification Report:
              precision    recall  f1-score   support

 Team 2 Wins       0.60      0.56      0.58       100
 Team 1 Wins       0.60      0.64      0.62       104

    accuracy                           0.60       204
   macro avg       0.60      0.60      0.60       204
weighted avg       0.60      0.60      0.60       204


======================================================================
MODEL COMPARISON SUMMARY
======================================================================

              Model  Train_Accuracy  Test_Accuracy  CV_Mean   CV_Std
      Random Forest        0.964286       0.607843 0.646580 0.027779
Logistic Regression        0.714286       0.617647 0.690873 0.014531
                SVM        0.724138       0.612745 0.658865 0.018041
K-Nearest Neighbors        1.000000       0.602941 0.607150 0.028872

======================================================================
BEST MODEL: Logistic Regression (Test Accuracy: 0.6176)
======================================================================

Creating comparison visualizations...
Saved: visualizations/model_comparison.png
Saved: visualizations/confusion_matrices.png

======================================================================
DETAILED MODEL METRICS
======================================================================

              Model  Accuracy  Precision (Team 1)  Recall (Team 1)  F1-Score (Team 1)  True Positives  False Positives  True Negatives  False Negatives
      Random Forest  0.607843            0.611111         0.634615           0.622642              66               42              58               38
Logistic Regression  0.617647            0.612069         0.682692           0.645455              71               45              55               33
                SVM  0.612745            0.612613         0.653846           0.632558              68               43              57               36
K-Nearest Neighbors  0.602941            0.603604         0.644231           0.623256              67               44              56               37

Results saved to datasets/

======================================================================
RECOMMENDATIONS
======================================================================

Based on the comparison:

1. BEST MODEL: Logistic Regression
   - Highest test accuracy: 61.76%
   - Good balance between bias and variance

2. OVERFITTING ANALYSIS:
   - Random Forest shows some overfitting (high train, moderate test)
   - Logistic Regression generalizes well (similar train/test)
   - SVM shows balanced performance

3. FOR PRODUCTION:
   - Use Logistic Regression for best accuracy
   - Consider ensemble of top 2-3 models for robustness

4. FUTURE IMPROVEMENTS:
   - Collect more data (especially recent matches)
   - Add player-level features
   - Try deep learning models
   - Implement stacking/ensemble methods


======================================================================
MODEL COMPARISON COMPLETE!
======================================================================

C:\Users\Asus\Documents\GitHub\Cricket World Cup Predictor\ICC-2026-WC-prediction>python predict_world_cup.py
Loading...
Model and data loaded successfully


====================================
ICC T20 WORLD CUP 2026 - PREDICTIONS
====================================

Group Compositions:

Group A:
  1. India                Rank:  1 | Rating: 722
  2. Namibia              Rank:  6 | Rating: 548
  3. Pakistan             Rank: 11 | Rating: 441
  4. Netherlands          Rank: 13 | Rating: 401
  5. United States Of America

Group B:
  1. Australia            Rank:  2 | Rating: 584
  2. Ireland              Rank: 15 | Rating: 365
  3. Oman                 Rank: 18 | Rating: 250
  4. Sri Lanka            Rank: 10 | Rating: 444
  5. Zimbabwe             Rank: 17 | Rating: 257

Group C:
  1. England              Rank:  5 | Rating: 554
  2. Italy                Rank: 20 | Rating: 0
  3. Nepal                Rank: 12 | Rating: 433
  4. Scotland             Rank: 14 | Rating: 366
  5. West Indies          Rank:  8 | Rating: 475

Group D:
  1. Afghanistan          Rank:  9 | Rating: 452
  2. Canada               Rank: 16 | Rating: 333
  3. New Zealand          Rank:  4 | Rating: 556
  4. South Africa         Rank:  7 | Rating: 491
  5. United Arab Emirates Rank: 21 | Rating: 0


=============================
GROUP STAGE MATCH PREDICTIONS
=============================

Group A Matches:
----------------------------------------------------------------------
India vs Namibia >>> India wins with 92.7% confidence
India vs Pakistan >>> India wins with 75.0% confidence
India vs Netherlands >>> India wins with 95.0% confidence
India vs United States Of America >>> India wins with 99.8% confidence
Namibia vs Pakistan >>> Pakistan wins with 70.9% confidence
Namibia vs Netherlands >>> Namibia wins with 67.2% confidence
Namibia vs United States Of America >>> Namibia wins with 97.4% confidence
Pakistan vs Netherlands >>> Pakistan wins with 77.8% confidence
Pakistan vs United States Of America >>> Pakistan wins with 99.1% confidence
Netherlands vs United States Of America >>> Netherlands wins with 95.6% confidence

Group B Matches:
----------------------------------------------------------------------
Australia vs Ireland >>> Australia wins with 80.8% confidence
Australia vs Oman >>> Australia wins with 96.4% confidence
Australia vs Sri Lanka >>> Australia wins with 53.9% confidence
Australia vs Zimbabwe >>> Australia wins with 86.9% confidence
Ireland vs Oman >>> Ireland wins with 76.6% confidence
Ireland vs Sri Lanka >>> Sri Lanka wins with 70.8% confidence
Ireland vs Zimbabwe >>> Zimbabwe wins with 50.7% confidence
Oman vs Sri Lanka >>> Sri Lanka wins with 91.6% confidence
Oman vs Zimbabwe >>> Zimbabwe wins with 73.0% confidence
Sri Lanka vs Zimbabwe >>> Sri Lanka wins with 69.3% confidence

Group C Matches:
----------------------------------------------------------------------
England vs Italy >>> England wins with 99.2% confidence
England vs Nepal >>> England wins with 88.9% confidence
England vs Scotland >>> England wins with 89.9% confidence
England vs West Indies >>> England wins with 56.6% confidence
Italy vs Nepal >>> Nepal wins with 89.2% confidence
Italy vs Scotland >>> Scotland wins with 85.0% confidence
Italy vs West Indies >>> West Indies wins with 98.1% confidence
Nepal vs Scotland >>> Nepal wins with 58.4% confidence
Nepal vs West Indies >>> West Indies wins with 83.2% confidence
Scotland vs West Indies >>> West Indies wins with 89.2% confidence

Group D Matches:
----------------------------------------------------------------------
Afghanistan vs Canada >>> Afghanistan wins with 77.4% confidence
Afghanistan vs New Zealand >>> New Zealand wins with 77.6% confidence
Afghanistan vs South Africa >>> South Africa wins with 65.8% confidence
Afghanistan vs United Arab Emirates >>> Afghanistan wins with 97.4% confidence
Canada vs New Zealand >>> New Zealand wins with 94.2% confidence
Canada vs South Africa >>> South Africa wins with 90.3% confidence
Canada vs United Arab Emirates >>> Canada wins with 91.1% confidence
New Zealand vs South Africa >>> New Zealand wins with 68.9% confidence
New Zealand vs United Arab Emirates >>> New Zealand wins with 99.4% confidence
South Africa vs United Arab Emirates >>> South Africa wins with 98.9% confidence


=========================
PREDICTED GROUP STANDINGS
=========================

Group A:
Team                     Played        Won       Lost     Points
----------------------------------------------------------------------
India                         4          4          0          8
Pakistan                      4          3          1          6
Namibia                       4          2          2          4
Netherlands                   4          1          3          2
United States Of America      4          0          4          0

Group B:
Team                     Played        Won       Lost     Points
----------------------------------------------------------------------
Australia                     4          4          0          8
Sri Lanka                     4          3          1          6
Zimbabwe                      4          2          2          4
Ireland                       4          1          3          2
Oman                          4          0          4          0

Group C:
Team                     Played        Won       Lost     Points
----------------------------------------------------------------------
England                       4          4          0          8
West Indies                   4          3          1          6
Nepal                         4          2          2          4
Scotland                      4          1          3          2
Italy                         4          0          4          0

Group D:
Team                     Played        Won       Lost     Points
----------------------------------------------------------------------
New Zealand                   4          4          0          8
South Africa                  4          3          1          6
Afghanistan                   4          2          2          4
Canada                        4          1          3          2
United Arab Emirates          4          0          4          0


===========================
QUALIFIED FOR SUPER 8 STAGE
===========================
Group A: India (1st), Pakistan (2nd)
Group B: Australia (1st), Sri Lanka (2nd)
Group C: England (1st), West Indies (2nd)
Group D: New Zealand (1st), South Africa (2nd)

==============
SUPER 8 GROUPS
==============

Super 8 Group 1: India, Sri Lanka, England, South Africa
Super 8 Group 2: Australia, Pakistan, New Zealand, West Indies

=========================
SUPER 8 STAGE PREDICTIONS
=========================

Super 8 Group 1:
----------------------------------------------------------------------
India vs Sri Lanka >>> India wins with 73.7% confidence
India vs England >>> India wins with 75.9% confidence
India vs South Africa >>> India wins with 79.5% confidence
Sri Lanka vs England >>> England wins with 58.6% confidence
Sri Lanka vs South Africa >>> South Africa wins with 51.0% confidence
England vs South Africa >>> England wins with 52.2% confidence

Super 8 Group 2:
----------------------------------------------------------------------
Australia vs Pakistan >>> Australia wins with 55.4% confidence
Australia vs New Zealand >>> New Zealand wins with 64.7% confidence
Australia vs West Indies >>> Australia wins with 58.2% confidence
Pakistan vs New Zealand >>> New Zealand wins with 73.6% confidence
Pakistan vs West Indies >>> West Indies wins with 69.2% confidence
New Zealand vs West Indies >>> West Indies wins with 51.3% confidence

=================
SUPER 8 STANDINGS
=================

Super 8 Group 1:
Team                          Played        Won       Lost     Points
----------------------------------------------------------------------
India                              3          3          0          6
England                            3          2          1          4
South Africa                       3          1          2          2
Sri Lanka                          3          0          3          0

Qualified for Semi-Finals: India, England

Super 8 Group 2:
Team                          Played        Won       Lost     Points
----------------------------------------------------------------------
Australia                          3          2          1          4
New Zealand                        3          2          1          4
West Indies                        3          2          1          4
Pakistan                           3          0          3          0

Qualified for Semi-Finals: Australia, New Zealand


=======================
SEMI-FINALS PREDICTIONS
=======================

MATCH PREDICTION: India vs New Zealand

India Statistics:
  Matches: 206
  Win Rate: 67.0%
  Recent Form: 75.0% (last 20 matches)

New Zealand Statistics:
  Matches: 195
  Win Rate: 54.4%
  Recent Form: 55.0% (last 20 matches)

Head-to-Head:
  Total matches: 22
  India wins: 12
  India win rate: 54.5%
  PREDICTION: India wins
  Confidence: 70.7%

MATCH PREDICTION: Australia vs England

Australia Statistics:
  Matches: 180
  Win Rate: 55.0%
  Recent Form: 60.0% (last 20 matches)

England Statistics:
  Matches: 174
  Win Rate: 54.0%
  Recent Form: 55.0% (last 20 matches)

Head-to-Head:
  Total matches: 21
  Australia wins: 10
  Australia win rate: 47.6%
  PREDICTION: Australia wins
  Confidence: 52.8%

================
FINAL PREDICTION
================

MATCH PREDICTION: India vs Australia

India Statistics:
  Matches: 206
  Win Rate: 67.0%
  Recent Form: 75.0% (last 20 matches)

Australia Statistics:
  Matches: 180
  Win Rate: 55.0%
  Recent Form: 60.0% (last 20 matches)

Head-to-Head:
  Total matches: 30
  India wins: 19
  India win rate: 63.3%
  PREDICTION: India wins
  Confidence: 61.5%

==============================================
ICC T20 WORLD CUP 2026 PREDICTED WINNER: INDIA
==============================================

Predictions saved to datasets/tournament_predictions.csv
```

---

## Notes

This project provides probabilistic predictions based on historical data. Actual tournament outcomes depend on many real-world factors such as player form, injuries, weather, and match conditions.

**Model Accuracy**: ~62%  
Use predictions as informed estimates, not certainties.

---

## License

MIT License | Copyright (c) 2026 Chanitha Disas Abeygunawardena
This project is intended for educational and research purposes only.
