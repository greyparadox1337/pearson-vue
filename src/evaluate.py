"""
UCS321: Dedicated Random Forest Diagnostic & Evaluation Suite
Dataset: student_habits_performance.csv.xlsx

Visualizations:
1. Random Forest Benchmark: Baseline vs Tuned (Accuracy, Macro F1, OOB Score, ROC-AUC)
2. Random Forest Confusion Matrix (Counts & Normalized Recall)
3. Multi-Class ROC Curves (One-vs-Rest)
4. Feature Importance Hierarchy (MDI Gini Impurity reduction)
5. Single Tree Estimator Extraction from the Random Forest Ensemble
6. Unit 4 PCA 2D Clustering Projection
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.tree import plot_tree
from sklearn.decomposition import PCA
import joblib

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import LABEL_MAPPING, INVERSE_LABEL_MAPPING, FEATURE_NAMES, load_excel_dataset

def generate_evaluation_visualizations(
    model_json="reports/model_comparison.json",
    predictions_csv="reports/test_set_predictions.csv",
    data_path="student_habits_performance.csv.xlsx",
    output_dir="reports/figures"
):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(model_json, 'r') as f:
        meta = json.load(f)
        
    models_data = meta['model_comparison']
    class_labels = list(LABEL_MAPPING.keys())
    
    # -------------------------------------------------------------
    # 1. Random Forest Optimization Benchmark: Baseline vs Tuned
    # -------------------------------------------------------------
    model_names = list(models_data.keys())
    accuracies = [models_data[m]['test_accuracy'] * 100 for m in model_names]
    f1_scores = [models_data[m]['test_f1_macro'] * 100 for m in model_names]
    oob_scores = [models_data[m]['oob_score'] * 100 for m in model_names]
    roc_aucs = [models_data[m]['test_roc_auc_ovr'] * 100 for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.20
    
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor='#ffffff')
    ax.set_facecolor('#f8fafc')
    
    rects1 = ax.bar(x - 1.5*width, accuracies, width, label='Test Accuracy (%)', color='#0284c7', edgecolor='#0369a1')
    rects2 = ax.bar(x - 0.5*width, f1_scores, width, label='Macro F1-Score (%)', color='#7c3aed', edgecolor='#6d28d9')
    rects3 = ax.bar(x + 0.5*width, oob_scores, width, label='Out-of-Bag (OOB) Score (%)', color='#f59e0b', edgecolor='#d97706')
    rects4 = ax.bar(x + 1.5*width, roc_aucs, width, label='ROC-AUC (OvR %)', color='#059669', edgecolor='#047857')
    
    ax.set_ylabel('Performance Metric (%)', color='#0f172a', fontsize=11, fontweight='bold')
    ax.set_title('Random Forest Optimization Benchmark: Baseline vs. Tuned\n(student_habits_performance.csv.xlsx)', color='#0f172a', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, color='#0f172a', fontsize=11, fontweight='600')
    ax.set_ylim(40, 100)
    ax.tick_params(colors='#0f172a')
    ax.grid(axis='y', linestyle='--', alpha=0.35, color='#94a3b8')
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', color='#0f172a', fontsize=8.5, fontweight='bold')
            
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    
    ax.legend(facecolor='#ffffff', edgecolor='#cbd5e1', labelcolor='#0f172a', fontsize=9.5)
    plt.tight_layout()
    comp_plot_path = os.path.join(output_dir, "model_comparison_benchmark.png")
    plt.savefig(comp_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {comp_plot_path}")
    
    # -------------------------------------------------------------
    # 2. Random Forest Confusion Matrix (Counts & Recall)
    # -------------------------------------------------------------
    champ_name = meta['champion_model']
    cm = np.array(models_data[champ_name]['confusion_matrix'])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor='#ffffff')
    for ax in [ax1, ax2]:
        ax.set_facecolor('#ffffff')
        ax.tick_params(colors='#0f172a')
        
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Student Count'})
    ax1.set_title(f'Random Forest Confusion Matrix (Counts)\n{champ_name}', color='#0f172a', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Predicted Performance', color='#0f172a', fontweight='bold')
    ax1.set_ylabel('Ground Truth Performance', color='#0f172a', fontweight='bold')
    
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', ax=ax2,
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Recall Proportion'})
    ax2.set_title(f'Random Forest Confusion Matrix (Recall)\n{champ_name}', color='#0f172a', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Predicted Performance', color='#0f172a', fontweight='bold')
    ax2.set_ylabel('Ground Truth Performance', color='#0f172a', fontweight='bold')
    
    plt.tight_layout()
    cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {cm_plot_path}")
    
    # -------------------------------------------------------------
    # 3. Random Forest Feature Importance (MDI Gini)
    # -------------------------------------------------------------
    importances_dict = meta.get('feature_importances', {})
    sorted_features = sorted(importances_dict.items(), key=lambda x: x[1], reverse=True)
    feat_names = [x[0].replace('_', ' ').title() for x in sorted_features]
    feat_scores = [x[1] * 100 for x in sorted_features]
    
    plt.figure(figsize=(10, 6), facecolor='#ffffff')
    ax = plt.subplot(111)
    ax.set_facecolor('#f8fafc')
    colors = sns.color_palette("viridis", len(feat_names))
    
    bars = ax.barh(feat_names[::-1], feat_scores[::-1], color=colors, edgecolor='#0f172a')
    ax.set_xlabel('Gini Importance / Feature Weight (%)', color='#0f172a', fontweight='bold', fontsize=11)
    ax.set_title('Random Forest Feature Importance (MDI) on student_habits_performance', color='#0f172a', fontweight='bold', fontsize=12, pad=12)
    ax.tick_params(colors='#0f172a')
    ax.grid(axis='x', linestyle='--', alpha=0.35, color='#94a3b8')
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.3, bar.get_y() + bar.get_height()/2, f'{w:.1f}%',
                ha='left', va='center', color='#0f172a', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    fi_plot_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(fi_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fi_plot_path}")
    
    # -------------------------------------------------------------
    # 4. Multi-Class ROC Curves (One-vs-Rest)
    # -------------------------------------------------------------
    pipeline = joblib.load("models/best_model_pipeline.joblib")
    test_df = pd.read_csv(predictions_csv)
    y_test_codes = test_df['ground_truth_code'].values
    y_test_bin = label_binarize(y_test_codes, classes=[0, 1, 2])
    
    raw_features_for_pred = test_df.drop(columns=['student_id', 'ground_truth_code', 'ground_truth_label', 'predicted_code', 'predicted_label'], errors='ignore')
    y_probs = pipeline.predict_proba(raw_features_for_pred)
    
    plt.figure(figsize=(8, 6.5), facecolor='#ffffff')
    ax = plt.subplot(111)
    ax.set_facecolor('#f8fafc')
    
    curve_colors = ['#dc2626', '#2563eb', '#16a34a']
    for i, class_label in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
        roc_score = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=curve_colors[i], lw=2.5,
                label=f'{class_label} (AUC = {roc_score:.3f})')
        
    ax.plot([0, 1], [0, 1], color='#64748b', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.50)')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', color='#0f172a', fontweight='bold', fontsize=11)
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', color='#0f172a', fontweight='bold', fontsize=11)
    ax.set_title(f'Random Forest Multi-Class ROC Curves (One-vs-Rest)', color='#0f172a', fontweight='bold', fontsize=12, pad=12)
    ax.tick_params(colors='#0f172a')
    ax.grid(True, linestyle='--', alpha=0.35, color='#94a3b8')
    ax.legend(loc="lower right", facecolor='#ffffff', edgecolor='#cbd5e1', labelcolor='#0f172a', fontsize=10)
    
    plt.tight_layout()
    roc_plot_path = os.path.join(output_dir, "roc_auc_curves.png")
    plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {roc_plot_path}")
    
    # -------------------------------------------------------------
    # 5. Single Tree Estimator Visualization from the Random Forest
    # -------------------------------------------------------------
    rf_clf = pipeline.named_steps['classifier']
    sample_tree = rf_clf.estimators_[0]
    
    plt.figure(figsize=(18, 8), facecolor='#ffffff')
    plot_tree(
        sample_tree, feature_names=FEATURE_NAMES, class_names=class_labels,
        max_depth=3, filled=True, rounded=True, fontsize=10
    )
    plt.title('Individual Decision Tree Sample from the Random Forest Ensemble (Estimator 0)', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    tree_plot_path = os.path.join(output_dir, "decision_tree_structure.png")
    plt.savefig(tree_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {tree_plot_path}")
    
    # -------------------------------------------------------------
    # 6. Unit 4 Connection: PCA 2D Dimensionality Reduction Visualization
    # -------------------------------------------------------------
    preprocessor = joblib.load("models/preprocessor.joblib")
    df_raw = load_excel_dataset(data_path)
    X_all_raw = df_raw.drop(columns=['performance_category', 'grade', 'total_score'], errors='ignore')
    y_all = df_raw['performance_category']
    
    X_scaled = preprocessor.transform(X_all_raw)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_
    
    plt.figure(figsize=(9, 6.5), facecolor='#ffffff')
    ax = plt.subplot(111)
    ax.set_facecolor('#f8fafc')
    
    pca_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Category': y_all
    })
    
    category_palette = {
        'High Performer': '#16a34a',
        'Average Performer': '#d97706',
        'Needs Improvement': '#dc2626'
    }
    
    for cat, color in category_palette.items():
        sub = pca_df[pca_df['Category'] == cat]
        ax.scatter(sub['PC1'], sub['PC2'], c=color, label=cat, alpha=0.6, s=35, edgecolors='none')
        
    ax.set_xlabel(f'Principal Component 1 ({var_exp[0]*100:.1f}% Variance)', color='#0f172a', fontweight='bold', fontsize=11)
    ax.set_ylabel(f'Principal Component 2 ({var_exp[1]*100:.1f}% Variance)', color='#0f172a', fontweight='bold', fontsize=11)
    ax.set_title('UCS321 Unit 4: PCA 2D Projection of student_habits_performance Data', color='#0f172a', fontweight='bold', fontsize=12, pad=12)
    ax.grid(True, linestyle='--', alpha=0.35, color='#94a3b8')
    ax.legend(facecolor='#ffffff', edgecolor='#cbd5e1', labelcolor='#0f172a', fontsize=10)
    
    plt.tight_layout()
    pca_plot_path = os.path.join(output_dir, "pca_clusters.png")
    plt.savefig(pca_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {pca_plot_path}")
    
    print("\nAll dedicated Random Forest visualizations generated successfully!")

if __name__ == "__main__":
    generate_evaluation_visualizations()
