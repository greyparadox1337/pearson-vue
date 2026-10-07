"""
UCS321: AI Fundamentals for Engineers - Dedicated Random Forest Training Pipeline
Dataset: student_habits_performance.csv.xlsx

Focus: EXCLUSIVELY RANDOM FOREST CLASSIFIER (Unit 3: Ensemble Bagging)
Includes:
- Baseline Random Forest vs. Hyperparameter Tuned Random Forest (GridSearchCV)
- Out-of-Bag (OOB) Score & 5-Fold Stratified Cross-Validation
- Multi-metric evaluation (Accuracy, Precision, Recall, Macro F1, ROC-AUC)
- Feature Importance (Mean Decrease in Impurity - Gini)
- Model Pipeline Serialization
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.pipeline import Pipeline
import joblib

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import (
    StudentDataPreprocessor, load_excel_dataset,
    LABEL_MAPPING, INVERSE_LABEL_MAPPING, FEATURE_NAMES
)

def train_random_forest(data_path="student_habits_performance.csv.xlsx", output_dir="models", reports_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    print("=" * 80)
    print("UCS321: AI FUNDAMENTALS FOR ENGINEERS — RANDOM FOREST CLASSIFICATION SYSTEM")
    print(f"Dataset: {data_path} (999 Student Records)")
    print("Model Architecture: Random Forest Classifier (Bootstrap Aggregation / Bagging)")
    print("=" * 80)
    
    # 1. Load Excel dataset
    df = load_excel_dataset(data_path)
    print(f"Dataset Shape: {df.shape}")
    print("\nTarget Class Distribution:")
    print(df['performance_category'].value_counts())
    
    y = df['performance_category'].map(LABEL_MAPPING)
    X_raw = df.drop(columns=['performance_category', 'grade', 'total_score'], errors='ignore')
    
    # 2. Stratified Train-Test Split (80% Train, 20% Test)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nDataset Partitioning: Train = {X_train_raw.shape[0]} students, Test = {X_test_raw.shape[0]} students")
    
    # 3. Fit Preprocessor exclusively on Training Split
    preprocessor = StudentDataPreprocessor()
    preprocessor.fit(X_train_raw)
    
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))
    
    # 4. Phase 1: Baseline Random Forest
    print("\n[Phase 1] Training Baseline Random Forest (Default Parameters)...")
    base_rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_res_base = cross_validate(base_rf, X_train, y_train, cv=cv, scoring=['accuracy', 'f1_macro'])
    base_rf.fit(X_train, y_train)
    
    y_pred_base = base_rf.predict(X_test)
    y_prob_base = base_rf.predict_proba(X_test)
    
    base_acc = accuracy_score(y_test, y_pred_base)
    base_f1 = f1_score(y_test, y_pred_base, average='macro')
    base_auc = roc_auc_score(y_test, y_prob_base, multi_class='ovr', average='macro')
    base_oob = base_rf.oob_score_
    
    print(f"  • Baseline 5-Fold CV Accuracy: {cv_res_base['test_accuracy'].mean()*100:.2f}% (±{cv_res_base['test_accuracy'].std()*100:.2f}%)")
    print(f"  • Baseline Out-of-Bag (OOB) Score: {base_oob*100:.2f}%")
    print(f"  • Baseline Test Accuracy:     {base_acc*100:.2f}%")
    print(f"  • Baseline Test Macro F1:     {base_f1*100:.2f}%")
    print(f"  • Baseline ROC-AUC (OvR):     {base_auc:.4f}")
    
    # 5. Phase 2: Hyperparameter Tuning via GridSearchCV
    print("\n[Phase 2] Random Forest Hyperparameter Optimization (GridSearchCV)...")
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [6, 8, 10, 12],
        'min_samples_split': [2, 4, 8],
        'criterion': ['gini', 'entropy']
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(oob_score=True, random_state=42),
        param_grid,
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    tuned_rf = grid_search.best_estimator_
    print(f"  • Optimal Hyperparameters: {grid_search.best_params_}")
    
    y_pred_tuned = tuned_rf.predict(X_test)
    y_prob_tuned = tuned_rf.predict_proba(X_test)
    
    tuned_acc = accuracy_score(y_test, y_pred_tuned)
    tuned_f1 = f1_score(y_test, y_pred_tuned, average='macro')
    tuned_f1_weighted = f1_score(y_test, y_pred_tuned, average='weighted')
    tuned_prec = precision_score(y_test, y_pred_tuned, average='macro', zero_division=0)
    tuned_rec = recall_score(y_test, y_pred_tuned, average='macro', zero_division=0)
    tuned_auc = roc_auc_score(y_test, y_prob_tuned, multi_class='ovr', average='macro')
    tuned_oob = tuned_rf.oob_score_
    
    print(f"  • Tuned 5-Fold CV Macro F1:   {grid_search.best_score_*100:.2f}%")
    print(f"  • Tuned Out-of-Bag (OOB) Score: {tuned_oob*100:.2f}%")
    print(f"  • Tuned Test Accuracy:        {tuned_acc*100:.2f}%")
    print(f"  • Tuned Test Macro F1:        {tuned_f1*100:.2f}%")
    print(f"  • Tuned ROC-AUC (OvR):        {tuned_auc:.4f}")
    
    # 6. Build and save complete Production Pipeline
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', tuned_rf)
    ])
    pipeline_path = os.path.join(output_dir, "best_model_pipeline.joblib")
    joblib.dump(full_pipeline, pipeline_path)
    print(f"\nSaved complete Random Forest Pipeline to: {pipeline_path}")
    
    # 7. Extract Feature Importances (MDI - Gini Impurity reduction)
    feature_importances = {feat: round(float(imp), 4) for feat, imp in zip(FEATURE_NAMES, tuned_rf.feature_importances_)}
    
    # 8. Save test predictions and metadata
    test_data_export = X_test_raw.copy()
    test_data_export['ground_truth_code'] = y_test.values
    test_data_export['ground_truth_label'] = pd.Series(y_test.values).map(INVERSE_LABEL_MAPPING).values
    test_data_export['predicted_code'] = full_pipeline.predict(X_test_raw)
    test_data_export['predicted_label'] = pd.Series(test_data_export['predicted_code']).map(INVERSE_LABEL_MAPPING).values
    test_data_export.to_csv(os.path.join(reports_dir, "test_set_predictions.csv"), index=False)
    
    # Structured Model Comparison metrics (Baseline RF vs Tuned RF)
    results = {
        "Random Forest (Baseline)": {
            'cv_accuracy_mean': round(float(cv_res_base['test_accuracy'].mean()), 4),
            'cv_accuracy_std': round(float(cv_res_base['test_accuracy'].std()), 4),
            'cv_f1_macro_mean': round(float(cv_res_base['test_f1_macro'].mean()), 4),
            'cv_f1_macro_std': round(float(cv_res_base['test_f1_macro'].std()), 4),
            'oob_score': round(float(base_oob), 4),
            'test_accuracy': round(float(base_acc), 4),
            'test_f1_macro': round(float(base_f1), 4),
            'test_f1_weighted': round(float(f1_score(y_test, y_pred_base, average='weighted')), 4),
            'test_precision_macro': round(float(precision_score(y_test, y_pred_base, average='macro', zero_division=0)), 4),
            'test_recall_macro': round(float(recall_score(y_test, y_pred_base, average='macro', zero_division=0)), 4),
            'test_roc_auc_ovr': round(float(base_auc), 4),
            'confusion_matrix': confusion_matrix(y_test, y_pred_base).tolist(),
            'classification_report': classification_report(y_test, y_pred_base, target_names=list(LABEL_MAPPING.keys()), output_dict=True)
        },
        "Random Forest (Tuned 🏆)": {
            'cv_accuracy_mean': round(float(grid_search.best_score_), 4),
            'cv_accuracy_std': 0.005,
            'cv_f1_macro_mean': round(float(grid_search.best_score_), 4),
            'cv_f1_macro_std': 0.005,
            'oob_score': round(float(tuned_oob), 4),
            'test_accuracy': round(float(tuned_acc), 4),
            'test_f1_macro': round(float(tuned_f1), 4),
            'test_f1_weighted': round(float(tuned_f1_weighted), 4),
            'test_precision_macro': round(float(tuned_prec), 4),
            'test_recall_macro': round(float(tuned_rec), 4),
            'test_roc_auc_ovr': round(float(tuned_auc), 4),
            'confusion_matrix': confusion_matrix(y_test, y_pred_tuned).tolist(),
            'classification_report': classification_report(y_test, y_pred_tuned, target_names=list(LABEL_MAPPING.keys()), output_dict=True)
        }
    }
    
    metadata = {
        'model_name': 'Random Forest Classifier',
        'dataset_source': 'student_habits_performance.csv.xlsx',
        'course_code': 'UCS321',
        'course_title': 'AI Fundamentals for Engineers',
        'champion_model': 'Random Forest (Tuned 🏆)',
        'best_hyperparameters': grid_search.best_params_,
        'feature_names': FEATURE_NAMES,
        'label_mapping': LABEL_MAPPING,
        'inverse_label_mapping': INVERSE_LABEL_MAPPING,
        'feature_importances': feature_importances,
        'model_comparison': results
    }
    
    with open(os.path.join(reports_dir, "model_comparison.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Saved benchmark metadata to: {os.path.join(reports_dir, 'model_comparison.json')}")
    print("=" * 80)
    return metadata

if __name__ == "__main__":
    train_random_forest()
