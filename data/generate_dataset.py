"""
Dataset Generator for Pearson VUE Student Performance Classification.
Generates realistic academic performance data with correlated signals,
realistic educational distributions, and missing values for preprocessing.
"""

import numpy as np
import pandas as pd
import os

def generate_student_dataset(n_samples=1500, random_state=42):
    np.random.seed(random_state)
    
    # 1. Latent academic engagement & aptitude variable (governs correlated features)
    latent_ability = np.random.normal(loc=0.0, scale=1.0, size=n_samples)
    
    # 2. Mid-Semester Test (MST) Score (0 to 100)
    # Strongly influenced by latent ability + test anxiety / variance
    mst_raw = 65 + (latent_ability * 18) + np.random.normal(0, 7, size=n_samples)
    mst_score = np.clip(mst_raw, 10, 100)
    
    # 3. Quizzes (Quiz 1, 2, 3 out of 20)
    quiz_1_raw = (65 + (latent_ability * 16) + np.random.normal(0, 6, size=n_samples)) / 5.0
    quiz_2_raw = (67 + (latent_ability * 17) + np.random.normal(0, 5.5, size=n_samples)) / 5.0
    quiz_3_raw = (64 + (latent_ability * 18) + np.random.normal(0, 6, size=n_samples)) / 5.0
    
    quiz_1 = np.round(np.clip(quiz_1_raw, 2, 20), 1)
    quiz_2 = np.round(np.clip(quiz_2_raw, 2, 20), 1)
    quiz_3 = np.round(np.clip(quiz_3_raw, 2, 20), 1)
    quiz_avg = np.round(((quiz_1 + quiz_2 + quiz_3) / 60.0) * 100.0, 1)
    
    # 4. Attendance Rate (0 to 100%)
    # Behavioral trait, correlated with diligence/engagement
    attendance_raw = 75 + (latent_ability * 14) + np.random.normal(0, 8, size=n_samples)
    attendance_rate = np.round(np.clip(attendance_raw, 25, 100), 1)
    
    # 5. Assignment Performance (0 to 100)
    assignment_raw = 70 + (latent_ability * 16) + np.random.normal(0, 7, size=n_samples)
    assignment_score = np.round(np.clip(assignment_raw, 20, 100), 1)
    
    # 6. Assignment Completion Rate (0 to 100%)
    completion_raw = 80 + (latent_ability * 12) + np.random.normal(0, 6, size=n_samples)
    assignment_completion_rate = np.round(np.clip(completion_raw, 30, 100), 1)
    
    # 7. Weekly Study Hours (5 to 35 hours)
    study_hours_raw = 14 + (latent_ability * 6) + np.random.normal(0, 3, size=n_samples)
    study_hours = np.round(np.clip(study_hours_raw, 2, 35), 1)
    
    # 8. True Composite Score with realistic noise for categorization
    # Realistic weights: MST 35%, Quizzes 25%, Assignments 25%, Attendance 15%
    composite = (
        0.35 * mst_score +
        0.25 * quiz_avg +
        0.25 * assignment_score +
        0.15 * attendance_rate +
        np.random.normal(0, 3.5, size=n_samples)
    )
    
    # 9. Classify into 3 categories:
    # High Performer: composite >= 73
    # Average Performer: 52 <= composite < 73
    # Needs Improvement: composite < 52
    categories = []
    for c in composite:
        if c >= 73.0:
            categories.append('High Performer')
        elif c >= 52.0:
            categories.append('Average Performer')
        else:
            categories.append('Needs Improvement')
            
    # Student IDs
    student_ids = [f"PV-{10000 + i}" for i in range(n_samples)]
    
    df = pd.DataFrame({
        'student_id': student_ids,
        'mst_score': np.round(mst_score, 1),
        'quiz_1': quiz_1,
        'quiz_2': quiz_2,
        'quiz_3': quiz_3,
        'quiz_avg': quiz_avg,
        'attendance_rate': attendance_rate,
        'assignment_score': assignment_score,
        'assignment_completion_rate': assignment_completion_rate,
        'study_hours_weekly': study_hours,
        'performance_category': categories
    })
    
    # Introduce deliberate missing values (MCAR/MAR) to demonstrate data preprocessing
    # ~3.5% missing in mst_score, ~4% in attendance_rate, ~3% in assignment_score, ~2% in quiz_avg
    missing_mask_mst = np.random.rand(n_samples) < 0.035
    missing_mask_att = np.random.rand(n_samples) < 0.04
    missing_mask_asg = np.random.rand(n_samples) < 0.03
    missing_mask_qavg = np.random.rand(n_samples) < 0.02
    
    df.loc[missing_mask_mst, 'mst_score'] = np.nan
    df.loc[missing_mask_att, 'attendance_rate'] = np.nan
    df.loc[missing_mask_asg, 'assignment_score'] = np.nan
    df.loc[missing_mask_qavg, 'quiz_avg'] = np.nan
    
    return df

def generate_test_samples(n_samples=25, random_state=999):
    np.random.seed(random_state)
    # Generate standalone test cohort without labels for batch inference demonstration
    df = generate_student_dataset(n_samples=n_samples, random_state=random_state)
    return df.drop(columns=['performance_category'])

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    print("Generating Pearson VUE Student Performance Benchmark Dataset...")
    dataset = generate_student_dataset(n_samples=1500, random_state=42)
    output_path = os.path.join("data", "student_performance_dataset.csv")
    dataset.to_csv(output_path, index=False)
    print(f"Dataset successfully saved to: {output_path}")
    print(f"Dataset Shape: {dataset.shape}")
    print("\nClass distribution:")
    print(dataset['performance_category'].value_counts())
    print("\nMissing values:")
    print(dataset.isna().sum())
    
    # Generate sample test batch
    test_batch = generate_test_samples(n_samples=30, random_state=777)
    test_batch_path = os.path.join("data", "test_samples.csv")
    test_batch.to_csv(test_batch_path, index=False)
    print(f"\nBatch test samples saved to: {test_batch_path}")
