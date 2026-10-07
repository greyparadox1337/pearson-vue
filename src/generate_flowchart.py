"""
Generates the Project Flow Diagram required by the marking rubric:
"Flow diagram must be given with Pre-processing and visualization steps clearly documented."
Dataset: student_habits_performance.csv.xlsx
Model: Random Forest Classifier (Dedicated Architecture)
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_flowchart(output_path="reports/figures/project_flowchart.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='#ffffff')
    ax.set_facecolor('#ffffff')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Define box styles
    box_blue = dict(boxstyle="round,pad=0.5,rounding_size=0.3", fc="#e0f2fe", ec="#0284c7", lw=2)
    box_green = dict(boxstyle="round,pad=0.5,rounding_size=0.3", fc="#dcfce7", ec="#16a34a", lw=2)
    box_purple = dict(boxstyle="round,pad=0.5,rounding_size=0.3", fc="#f3e8ff", ec="#9333ea", lw=2)
    box_orange = dict(boxstyle="round,pad=0.5,rounding_size=0.3", fc="#ffedd5", ec="#ea580c", lw=2)
    box_amber = dict(boxstyle="round,pad=0.5,rounding_size=0.3", fc="#fef3c7", ec="#d97706", lw=2)
    box_gray = dict(boxstyle="round,pad=0.5,rounding_size=0.3", fc="#f1f5f9", ec="#475569", lw=2)
    
    # Title
    ax.text(50, 96, "UCS321: AI Fundamentals for Engineers — Random Forest Flow Diagram",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#0f172a")
    ax.text(50, 93, "Dataset: student_habits_performance.csv.xlsx | Architecture: Random Forest Classifier",
            ha="center", va="center", fontsize=11, color="#475569")
    
    # Step 1: Data Collection & Ingestion
    ax.text(50, 86, "1. DATA INGESTION & DATASET EXPLORATION\n• Dataset: student_habits_performance.csv.xlsx (999 Records)\n• Input Predictors: exam_score (MST), attendance, class_participation, study hours, sleep, screen time\n• Target: grade (A, B, C, D, F) mapped to (High Performer, Average Performer, Needs Improvement)",
            ha="center", va="center", fontsize=10, bbox=box_blue, multialignment="center")
    
    # Arrow 1 -> 2
    ax.annotate('', xy=(50, 78), xytext=(50, 81),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#0284c7"))
    
    # Step 2: Exploratory Data Analysis & Visualizations
    ax.text(50, 74, "2. EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATIONS\n• Class Balance Inspection (561 High Performer, 257 Average, 181 Needs Improvement)\n• Distribution Boxplots across Exam Score, Study Hours, and Sleep Duration\n• Correlation Matrix Heatmap (sns.heatmap) to identify multi-collinearity",
            ha="center", va="center", fontsize=10, bbox=box_green, multialignment="center")
    
    # Arrow 2 -> 3
    ax.annotate('', xy=(50, 66), xytext=(50, 69),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#16a34a"))
    
    # Step 3: Data Preprocessing (Crucial 10 Marks)
    ax.text(50, 58, "3. DATA PREPROCESSING (UCS321 Unit 2: Data Cleaning)\n• Dropped unformatted empty Excel columns (Unnamed: 16, 17, 18)\n• Missing Value Imputation: SimpleImputer (Mode for parental_education_level, Median for numericals)\n• Target Leakage Prevention: Excluded total_score (direct post-hoc grade formula)\n• Feature Normalization: StandardScaler (Zero Mean, Unit Variance)",
            ha="center", va="center", fontsize=10, bbox=box_orange, multialignment="center")
    
    # Arrow 3 -> 4
    ax.annotate('', xy=(50, 49), xytext=(50, 52),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#ea580c"))
    
    # Step 4: Dataset Partitioning
    ax.text(50, 45, "4. DATASET PARTITIONING (Anti-Leakage Protocol)\n• Stratified Train-Test Split (80% Train = 799 | 20% Test = 200 samples)\n• Imputer & Scaler fitted strictly on Training split to ensure unbiased holdout",
            ha="center", va="center", fontsize=10, bbox=box_gray, multialignment="center")
    
    # Arrow 4 -> 5
    ax.annotate('', xy=(50, 37.5), xytext=(50, 41),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#475569"))
    
    # Step 5: Random Forest Architecture (Unit 3 Core Syllabus)
    ax.text(50, 31, "5. RANDOM FOREST MODEL DEVELOPMENT (Unit 3: Ensemble Bagging)\n• Bootstrap Aggregation: Trains B independent decision trees on random data subsets with replacement\n• Random Feature Subsets: Evaluates m = sqrt(p) random features at each split to decorrelate trees\n• Hyperparameter Optimization: GridSearchCV over n_estimators, max_depth, min_samples_split, criterion\n• Validation: Out-of-Bag (OOB) error estimation + 5-Fold Stratified Cross-Validation",
            ha="center", va="center", fontsize=10, bbox=box_purple, multialignment="center")
    
    # Arrow 5 -> 6
    ax.annotate('', xy=(50, 22), xytext=(50, 25),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#9333ea"))
    
    # Step 6: Performance Evaluation
    ax.text(50, 16.5, "6. PERFORMANCE EVALUATION & INTERPRETATION (8 Marks)\n• Confusion Matrix Heatmaps (Raw student counts & Normalized recall rates)\n• Comprehensive Metrics: Precision, Recall, Macro F1-Score, and Overall Accuracy\n• Multi-Class ROC Curves (One-vs-Rest) & Area Under Curve (AUC)\n• Feature Importance: Mean Decrease in Impurity (Gini MDI Weight Attribution)",
            ha="center", va="center", fontsize=10, bbox=box_amber, multialignment="center")
    
    # Arrow 6 -> 7
    ax.annotate('', xy=(50, 9.5), xytext=(50, 12),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#d97706"))
    
    # Step 7: Innovation & Deployment
    ax.text(50, 5.5, "7. INNOVATION & EARLY WARNING PRESCRIBER (5 Marks)\n• Unit 4 PCA 2D Clustering: Unsupervised projection confirming student performance clusters\n• Automated Educational Action Plans: Remedial Counseling, Study Hall, Screen-Time Management\n• Local Web Dashboard & Batch CSV Scoring Tool (Localhost:5001)",
            ha="center", va="center", fontsize=10, bbox=box_green, multialignment="center")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Flowchart successfully regenerated at: {output_path}")

if __name__ == "__main__":
    generate_flowchart()
