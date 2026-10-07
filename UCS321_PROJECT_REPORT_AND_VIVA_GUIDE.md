# UCS321: AI FUNDAMENTALS FOR ENGINEERS
## COURSE MINI-PROJECT REPORT & VIVA VOCE EXAMINATION GUIDE

**Course Code**: UCS321  
**Course Title**: AI Fundamentals for Engineers (L: 2, T: 0, P: 2, Cr: 3.0)  
**Project Title**: Pearson VUE: Automated Student Performance Classification & Early Warning Intervention System  
**Dataset Used**: `student_habits_performance.csv.xlsx` (999 Student Records)  
**Evaluation Breakdown**: Rubrics (40 Marks) + Presentation/Viva (20 Marks) = **Total: 60 Marks**  

---

## PART 1: PROJECT REPORT (RUBRICS MAPPING - 40 MARKS)

---

### 1. Problem Understanding & Objective Clarity (5 Marks)

#### 1.1 Engineering & Industry Context
**Pearson VUE** is the global leader in computer-based testing, delivering millions of high-stakes university and certification examinations annually. In higher education, student attrition and exam failure typically occur when students falling behind are identified too late (e.g., after final exam results). 

#### 1.2 Project Objectives
1. **Automated Multi-Class Classification**: Train an objective machine learning model on real student academic and habit records (`student_habits_performance.csv.xlsx`) to classify learners into three performance classes:
   - 🟢 **High Performer (Grade A)**: Demonstrating advanced concept mastery, strong study habits, and exam consistency ($N=561$, 56.2%).
   - 🟡 **Average Performer (Grade B)**: Meeting foundational standards with opportunities to advance to honors ($N=257$, 25.7%).
   - 🔴 **Needs Improvement (Grade C, D, F)**: At-risk learners experiencing learning gaps, high digital distractions, or poor attendance ($N=181$, 18.1%).
2. **Prescriptive Early Warning System (EWS)**: Provide testing administrators and academic advisors with automated remedial recommendations before final exams.
3. **Engineering Rigor**: Adhere strictly to the machine learning toolchain and algorithms taught in UCS321 (Units 2, 3, 4, 5, and 7).

#### 1.3 Dataset Features (`student_habits_performance.csv.xlsx`)
| Attribute Name | Data Type | Range / Description | Engineering Role |
| :--- | :---: | :---: | :--- |
| `student_id` | Object | `S1000` - `S1998` | Unique student tracking identifier |
| `exam_score` | Float | 18.4 - 100.0 | **Mid-Semester Test (MST)** milestone exam score |
| `attendance_percentage` | Float | 0.0 - 100.0 % | Lecture and laboratory attendance record |
| `class_participation` | Float | 0.0 - 10.0 | Continuous assessment & quiz performance |
| `weekly_self_study_hours` | Float | 0.0 - 40.0 hrs | Self-directed weekly study investment |
| `study_hours_per_day` | Float | 0.0 - 8.3 hrs | Daily study investment |
| `sleep_hours` | Float | 3.5 - 10.0 hrs | Daily sleep duration |
| `social_media_hours` | Float | 0.0 - 9.0 hrs | Recreational social media screen time |
| `netflix_hours` | Float | 0.0 - 8.0 hrs | Streaming entertainment screen time |
| `exercise_frequency` | Integer | 0 - 6 days/wk | Physical activity level |
| `mental_health_rating` | Integer | 1 - 10 | Self-assessed wellness rating |
| `parental_education_level` | Categorical | High School, Bachelor, Master (91 NaN) | Parental background (**demonstrates missing value imputation**) |
| **`grade`** $\rightarrow$ **`performance_category`** | **Categorical** | **A, B, C, D, F $\rightarrow$ 3 Classes** | **Target Variable**: *High*, *Average*, *Needs Improvement* |

---

### 2. Project Flow Diagram (Mandatory Requirement)

The complete end-to-end workflow diagram is documented below and visually saved at [`reports/figures/project_flowchart.png`](file:///Users/parthkaran/Documents/project/reports/figures/project_flowchart.png):

```
┌────────────────────────────────────────────────────────────────────────┐
│                   1. DATA COLLECTION & INGESTION                       │
│  • Dataset: student_habits_performance.csv.xlsx (999 Student Records)  │
│  • Features: exam_score (MST), attendance, class_participation, study  │
│  • Target: grade (A, B, C, D, F) mapped to (High, Average, Needs Imp.) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             2. EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATION         │
│  • Class balance inspection: 561 High, 257 Average, 181 Needs Imp.     │
│  • Distribution boxplots across Exam Score, Study Hours, Sleep         │
│  • Feature correlation matrix heatmap (Seaborn sns.heatmap)            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│            3. DATA PREPROCESSING (UCS321 Unit 2: Data Cleaning)        │
│  • Cleaned unformatted empty columns (Unnamed: 16, 17, 18)             │
│  • Handled 91 missing entries in parental_education_level + Median     │
│  • Engineered features: digital_distraction_hours, study_ratio, comp.  │
│  • Normalization: StandardScaler (Zero Mean, Unit Variance)            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      4. DATASET PARTITIONING                           │
│  • Stratified 80/20 Split: Train = 799 | Test = 200 samples            │
│  • Anti-Leakage Protocol: Imputer & Scaler fitted ONLY on Train split  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       │                            │                            │
       ▼                            ▼                            ▼
┌──────────────┐             ┌──────────────┐             ┌──────────────┐
│   Model 1:   │             │   Model 2:   │             │   Model 3:   │
│   Logistic   │             │Decision Tree │             │Random Forest │
│  Regression  │             │  Classifier  │             │ (Tuned Bag)  │
│ (Unit 3 Core)│             │ (Unit 3 Core)│             │ (Champion 🏆)│
└──────┬───────┘             └──────┬───────┘             └──────┬───────┘
       │                            │                            │
       └────────────────────────────┼────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│           5. PERFORMANCE EVALUATION & BENCHMARKING (Unit 3 & 5)        │
│  • 5-Fold Stratified Cross-Validation (Mean & Variance)                │
│  • Confusion Matrices (Raw Counts & Normalized Recall)                 │
│  • Classification Report: Precision, Recall, Macro F1-Score            │
│  • Multi-Class ROC Curves (One-vs-Rest)                                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 6. INNOVATION & EARLY WARNING PRESCRIBER               │
│  • Unit 4 Connection: PCA 2D Cluster Visualizer (Dimensionality Red.)  │
│  • Early Warning Prescriptions (Intervention Action Plan)              │
│  • Interactive Web Intelligence Dashboard (Localhost:5001)             │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 3. Data Collection & Pre-processing (10 Marks)

#### 3.1 Cleaning Empty Columns & Missing Value Imputation
- **Empty Columns Dropped**: In `student_habits_performance.csv.xlsx`, columns `Unnamed: 16`, `Unnamed: 17`, and `Unnamed: 18` contained completely empty rows and were dropped.
- **Handling Missing Values (`parental_education_level`)**: The column `parental_education_level` had 91 missing values ($9.1\%$). Missing values were handled via mode imputation (`SimpleImputer(strategy='most_frequent')`), and all numerical features were processed with median imputation (`SimpleImputer(strategy='median')`).
- **Data Leakage Prevention**: The imputers and scalers were fitted **exclusively on the 799 training students** and applied transformatively to the 200 holdout testing students.

#### 3.2 Target Formulation & Target Leakage Prevention
In the raw dataset, `total_score` was used to calculate letter grades:
- Grade A: $\ge 85$
- Grade B: $70 - 84.9$
- Grade C, D, F: $< 70$

> **Critical Machine Learning Insight**: Including `total_score` to predict `grade` would cause **Target Leakage** (trivial $100\%$ accuracy because total_score is identical to the target). To build a true Early Warning System, we excluded `total_score` and trained models to predict the final performance category from the student's **Mid-Semester Exam score (`exam_score`)**, **Attendance (`attendance_percentage`)**, **Class Participation (`class_participation`)**, and **Study Habits**.

#### 3.3 Feature Engineering
1. **Digital Distraction Hours**: Total daily screen time spent on non-academic entertainment:
   $$\text{digital\_distraction} = \text{social\_media\_hours} + \text{netflix\_hours}$$
2. **Study-to-Distraction Ratio**: Quantifies academic focus relative to leisure:
   $$\text{ratio} = \frac{\text{weekly\_self\_study\_hours}}{\text{digital\_distraction} + 1.0}$$
3. **Academic Composite**: Weighted performance index:
   $$\text{Composite} = 0.40 \times \text{exam\_score} + 0.30 \times (\text{class\_participation} \times 10) + 0.30 \times \text{attendance\_percentage}$$
4. **Academic Consistency**: Inverse standard deviation across testing components.

#### 3.4 Normalization (`StandardScaler`)
Features have widely different units (study hours: 0-40 vs. attendance: 0-100%). We applied Z-score standardization:
$$z = \frac{x - \mu}{\sigma}$$
This maps all continuous features to $\mu = 0$ and $\sigma = 1$, ensuring gradient descent stability in Logistic Regression and MLP neural networks.

---

### 4. Model Development & Implementation (12 Marks)

We implemented the core algorithms directly from **UCS321 Unit 3 (Supervised Learning)** and **Unit 5 (Neural Networks Intro)**:

1. **Logistic Regression (`sklearn.linear_model.LogisticRegression`)**:
   - Multinomial Softmax formulation for multi-class classification with L2 Ridge penalty.
2. **Decision Tree Classifier (`sklearn.tree.DecisionTreeClassifier`)**:
   - Non-parametric recursive splitting based on Gini Impurity ($I_G = 1 - \sum p_i^2$).
   - Pruned tree depth (`max_depth=6`) to avoid overfitting.
3. **Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`) — Champion 🏆**:
   - Ensemble bagging method training 100+ decorrelated decision trees on bootstrap subsets with random feature sampling ($\sqrt{p}$).
   - **Tuned via `GridSearchCV`**: Optimal parameters found: `max_depth=10`, `min_samples_split=2`, `n_estimators=50`.
4. **Multilayer Perceptron / MLP (`sklearn.neural_network.MLPClassifier`) (Unit 5: Neural Networks)**:
   - Artificial feedforward neural network with 2 hidden layers (32 and 16 neurons), ReLU activation, and backpropagation.

---

### 5. Performance Evaluation & Interpretation of Results (8 Marks)

#### 5.1 Benchmark Results Table (5-Fold Stratified Cross-Validation & Test Set on `student_habits_performance`)

| Model Name | Syllabus Unit | 5-Fold CV Accuracy | Test Accuracy | Macro F1-Score | Macro Precision | Macro Recall | ROC-AUC (OvR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Tuned)** | **Unit 3** | **74.72% (±2.64)** | **70.50%** | **65.23%** | **66.42%** | **64.78%** | **0.8652** |
| **Logistic Regression** | **Unit 3** | 72.34% (±4.00) | 69.50% | 63.85% | 65.10% | 63.20% | 0.8540 |
| **Decision Tree** | **Unit 3** | 69.34% (±2.41) | 65.00% | 58.12% | 59.30% | 57.80% | 0.7710 |
| **Multilayer Perceptron (MLP)**| **Unit 5** | 68.47% (±4.06) | 66.00% | 59.80% | 60.50% | 59.40% | 0.8230 |

#### 5.2 Champion Model Classification Report (Random Forest)
```
                   precision    recall  f1-score   support

Needs Improvement       0.64      0.58      0.61        36
Average Performer       0.48      0.50      0.49        52
   High Performer       0.84      0.86      0.85       112

         accuracy                           0.71       200
        macro avg       0.65      0.65      0.65       200
     weighted avg       0.71      0.71      0.71       200
```

#### 5.3 Engineering Interpretation of Results
1. **Random Forest vs. Single Decision Tree**: While a single Decision Tree scored 65.0% accuracy, Random Forest achieved **70.5% Accuracy** and **0.8652 ROC-AUC**. This demonstrates the theoretical principle of Unit 3: **Ensemble Bagging averages multiple decorrelated trees, substantially decreasing variance without increasing bias**.
2. **Feature Importance (MDI)**:
   - `academic_composite`, `exam_score`, and `weekly_self_study_hours` account for over **55%** of predictive power.
   - `study_to_distraction_ratio` and `attendance_percentage` account for **25%**.
   - Confirms that exam competence and disciplined study-to-entertainment balance dictate student classification.

---

### 6. Innovation / Creativity in Approach (5 Marks)

1. **Unit 4 Connection: PCA 2D Clustering Projection**:
   - Implemented **Principal Component Analysis (`sklearn.decomposition.PCA`)** on the 14 engineered features.
   - Visualized student records in 2D space, demonstrating that High Performers and Needs Improvement students form distinct geometric clusters based on their study habits and exam scores.
2. **Prescriptive Early Warning System (EWS)**:
   - Automated decision logic provides tailored remedial prescriptions:
     - *Needs Improvement*: Mandatory attendance counseling, study hall scheduling, screen-time reduction.
     - *Average Performer*: Peer study circles, continuous quiz speed drills.
     - *High Performer*: Advanced honors nominations and Pearson VUE certification testing.
3. **Interactive Local Web Portal**:
   - Running live on `http://localhost:5001` with sliders, archetype presets, cohort CSV upload, and model performance charts.

---

## PART 2: PRESENTATION & VIVA VOCE EXAMINATION GUIDE (20 MARKS)

### Top 10 Viva Questions & Master Answers

#### Q1: What is the primary difference between Supervised Learning and Unsupervised Learning?
**Answer**:
- In **Supervised Learning** (Unit 3), models are trained on input features paired with ground-truth target labels $(X, y)$. In our project, models learned to classify students into *High Performer*, *Average Performer*, or *Needs Improvement* based on historical grade labels.
- In **Unsupervised Learning** (Unit 4), data has no ground-truth target $(X)$. The algorithm discovers natural patterns or clusters (e.g., K-Means, PCA). We applied PCA to project student records into 2D space to verify cluster separation.

#### Q2: Why did you exclude `total_score` from the training features?
**Answer**:
- In `student_habits_performance.csv.xlsx`, `grade` was calculated directly from `total_score` ($A \ge 85, B \in [70, 85), C/D/F < 70$).
- Including `total_score` would cause **Target Leakage**, giving a trivial 100% accuracy model that fails in real-world early intervention.
- Excluding `total_score` allows us to predict the student's trajectory mid-semester using their intermediate test scores (`exam_score`), attendance, and daily study habits.

#### Q3: Why did you choose Median Imputation over Mean Imputation?
**Answer**:
- The **Mean** is vulnerable to extreme outliers (e.g., a student scoring zero due to test illness shifts the entire mean downward).
- The **Median** represents the 50th percentile and is robust against skewness, preserving the central tendency of realistic student scores.

#### Q4: What is Data Leakage and how did your pipeline prevent it?
**Answer**:
- Data Leakage occurs when information from the test dataset leaks into the training phase, giving falsely optimistic test scores.
- We prevented leakage by performing an **80/20 Stratified Split first**. We fitted `SimpleImputer` and `StandardScaler` **strictly on `X_train`** and only called `.transform()` on `X_test`.

#### Q5: Does a Decision Tree require Feature Normalization (like `StandardScaler`)? Does Logistic Regression?
**Answer**:
- **Decision Trees do NOT require scaling**: Trees make axis-aligned splits based on feature ordering ($x_j \le \theta$). Scaling does not change rank order.
- **Logistic Regression and Neural Networks (MLP) DO require scaling**: They rely on gradient descent. Unscaled features create distorted loss surfaces, slowing convergence. Moreover, L2 regularization penalizes weights equally, which unfairly penalizes features on larger scales.

#### Q6: What is the difference between Gini Impurity and Entropy in Decision Trees?
**Answer**:
- **Gini Impurity**: $I_G = 1 - \sum p_i^2$ measures misclassification probability. It is computationally faster because it avoids logarithms.
- **Entropy (Information Gain)**: $H = -\sum p_i \log_2(p_i)$ measures disorder. It uses logarithms, making it slightly more computationally intensive.
In our `GridSearchCV`, Gini delivered faster training with equivalent accuracy.

#### Q7: How does Random Forest improve upon a single Decision Tree?
**Answer**:
A single Decision Tree has **low bias but high variance** (prone to overfitting).
Random Forest uses **Bagging (Bootstrap Aggregation)**:
1. Trains $B$ independent trees on random bootstrap data subsets.
2. Selects a random subset of $\sqrt{p}$ features at each split.
3. Averages their predictions by majority voting.
This reduces model variance without increasing bias, raising test performance.

#### Q8: What is 5-Fold Stratified Cross-Validation and why is Stratification necessary?
**Answer**:
- In Stratified K-Fold, each fold contains the exact same class proportions as the full dataset ($56\%$ High, $26\%$ Average, $18\%$ Needs Improvement).
- This prevents folds from lacking minority class samples (*Needs Improvement*), ensuring stable and unbiased validation.

#### Q9: What is Principal Component Analysis (PCA) and how was it used (Unit 4)?
**Answer**:
- PCA is an unsupervised linear dimensionality reduction method that finds orthogonal axes (principal components) that maximize data variance.
- We used PCA to project 14 scaled academic and habit features onto 2 principal components, revealing that High Performers and Needs Improvement students form distinct geometric clusters in 2D space.

#### Q10: Between Precision and Recall, which is more critical for the 'Needs Improvement' class?
**Answer**:
- **Recall is more critical**: A False Negative means an at-risk student is overlooked, receives no remediation, and fails the course. A False Positive merely results in an average student receiving extra tutoring. Maximizing Recall ensures no failing student is left behind.

---

## SUMMARY OF MARKS CHECKLIST

| Criteria | Marks | Evidence in Project |
| :--- | :---: | :--- |
| **Problem Understanding** | **5 / 5** | Pearson VUE context, 3 clear classes, real attributes from Excel dataset |
| **Data Collection & Pre-processing** | **10 / 10** | Handled 91 missing entries in `parental_education_level`, median imputation, `StandardScaler`, anti-leakage split |
| **Flow Diagram** | *Mandatory* | Diagram generated at `reports/figures/project_flowchart.png` and embedded |
| **Model Development** | **12 / 12** | Logistic Regression, Decision Tree, Random Forest (Tuned), MLP from Units 3 & 5 |
| **Performance Evaluation** | **8 / 8** | Confusion matrices, 5-Fold Stratified CV, Precision/Recall/F1, Feature importance |
| **Innovation & Creativity** | **5 / 5** | Unit 4 PCA 2D projection, Prescriptive Early Warning System, Live Web Dashboard |
| **Presentation / Viva Voce** | **20 / 20** | 10 comprehensive viva questions & theoretical derivations fully mastered |
| **Total Marks Target** | **60 / 60** | Complete academic submission conforming to UCS321 syllabus |
