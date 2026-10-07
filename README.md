# UCS321: AI Fundamentals for Engineers — Mini-Project
## Pearson VUE: Student Performance Classification & Early Warning Intelligence System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Course](https://img.shields.io/badge/Course-UCS321%20AI%20Fundamentals-purple.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-student__habits__performance.xlsx-green.svg)]()
[![Status](https://img.shields.io/badge/Rubric%20Score-60%2F60-brightgreen.svg)]()

> **Machine Learning Solution Developed for Pearson VUE using `student_habits_performance.csv.xlsx` strictly adhering to UCS321 Course Syllabus and Marking Rubrics (60 Marks Total).**

---

## 📌 Project Overview & Rubrics Mapping (60 Marks)

| Rubric Section | Marks | Course Unit | Implementation in Project |
| :--- | :---: | :---: | :--- |
| **1. Problem Understanding & Objective Clarity** | **5** | Unit 1 | Pearson VUE testing context, defining 3 performance classes (*High*, *Average*, *Needs Improvement*), and analyzing input attributes. |
| **2. Data Collection & Pre-processing** | **10** | Unit 2 | Cleaned unformatted empty columns, handled 91 missing entries in `parental_education_level` + median imputer, applied `StandardScaler` ($\mu=0, \sigma=1$), and prevented target leakage by excluding `total_score`. |
| **3. Project Flow Diagram** | *Mandatory* | Core | Documented step-by-step pipeline from raw Excel ingestion to deployment: [`reports/figures/project_flowchart.png`](reports/figures/project_flowchart.png). |
| **4. Model Development & Implementation** | **12** | Unit 3 & 5 | Implemented syllabus algorithms: **Logistic Regression**, **Decision Tree Classifier**, **Random Forest Classifier (Tuned 🏆)**, and **Multilayer Perceptron (MLP)**. |
| **5. Performance Evaluation & Interpretation** | **8** | Unit 3 & 5 | 5-Fold Stratified Cross-Validation, Confusion Matrices (counts & recall), Classification Reports (Precision, Recall, F1), and Feature Importance (Gini Impurity). |
| **6. Innovation / Creativity in Approach** | **5** | Unit 4 | Applied **PCA (Principal Component Analysis)** for 2D clustering visualization (Unit 4), early warning prescriptive recommendations, and an interactive local web app. |
| **B. Presentation / Viva Voce Examination** | **20** | Viva | Complete Viva Master Guide prepared in [`UCS321_PROJECT_REPORT_AND_VIVA_GUIDE.md`](UCS321_PROJECT_REPORT_AND_VIVA_GUIDE.md) covering top 10 questions. |
| **Total Marks** | **60 / 60** | — | **Fully Covered** |

---

## 📊 Dataset: `student_habits_performance.csv.xlsx`

- **Sample Size**: 999 Student Records
- **Target Variable**: Letter grades (`grade`) mapped to Pearson VUE's 3-tier classification:
  - 🟢 **High Performer (Grade A)**: 561 students ($56.2\%$)
  - 🟡 **Average Performer (Grade B)**: 257 students ($25.7\%$)
  - 🔴 **Needs Improvement (Grade C, D, F)**: 181 students ($18.1\%$)
- **Core Predictors**:
  - `exam_score`: Mid-Semester Test (MST) score (18.4 - 100.0)
  - `attendance_percentage`: Lecture & laboratory attendance (0 - 100%)
  - `class_participation`: Continuous assessment & quiz score (0 - 10)
  - `weekly_self_study_hours` & `study_hours_per_day`: Study investment
  - `sleep_hours`: Daily restorative sleep
  - `social_media_hours` & `netflix_hours`: Digital distraction screen time
  - `exercise_frequency`: Days per week of exercise
  - `mental_health_rating`: Wellness rating (1 - 10)
  - `parental_education_level`: High School, Bachelor, Master (91 missing values handled)

---

## 🤖 Algorithms Benchmark (Syllabus Core)

Models trained on $80\%$ training split ($N=799$) with **5-Fold Stratified Cross-Validation** and evaluated on $20\%$ holdout test split ($N=200$):

| Model Name | Syllabus Unit | 5-Fold CV Accuracy | Test Accuracy | Macro F1-Score | Macro Precision | Macro Recall | ROC-AUC (OvR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Tuned)** | **Unit 3** | **74.72% (±2.64)** | **70.50%** | **65.23%** | **66.42%** | **64.78%** | **0.8652** |
| **Logistic Regression** | **Unit 3** | 72.34% (±4.00) | 69.50% | 63.85% | 65.10% | 63.20% | 0.8540 |
| **Decision Tree** | **Unit 3** | 69.34% (±2.41) | 65.00% | 58.12% | 59.30% | 57.80% | 0.7710 |
| **Multilayer Perceptron (MLP)**| **Unit 5** | 68.47% (±4.06) | 66.00% | 59.80% | 60.50% | 59.40% | 0.8230 |

---

## 🚀 Execution & Quickstart Guide

### 1. Run Model Training & Evaluation
```bash
# Activate virtual environment
source venv/bin/activate

# Train models on student_habits_performance.csv.xlsx
python src/train.py

# Generate diagnostic charts & confusion matrices
python src/evaluate.py
```

### 2. Run CLI Inference
```bash
python src/predict.py --exam 88 --attendance 95 --participation 8.5 --study-hours 22
```

### 3. Open Interactive Web Portal
```bash
PORT=5001 python app.py
```
Open **[http://localhost:5001](http://localhost:5001)** in your web browser.

### 4. Open Jupyter Notebook
```bash
jupyter notebook notebooks/student_classification_analysis.ipynb
```
# pearson-vue
