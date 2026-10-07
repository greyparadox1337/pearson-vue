"""
UCS321: AI Fundamentals for Engineers - Preprocessing Pipeline
Trained on: student_habits_performance.csv.xlsx

Handles:
- Loading Excel dataset and dropping empty columns (Unnamed)
- Mapping student letter grades ('A', 'B', 'C', 'D', 'F') to Pearson VUE Performance Categories:
  - 'A' -> 'High Performer'
  - 'B' -> 'Average Performer'
  - 'C', 'D', 'F' -> 'Needs Improvement'
- Imputing missing values (parental_education_level, numerical medians)
- Engineering academic composite and lifestyle interaction features
- Normalization via StandardScaler
"""

import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

LABEL_MAPPING = {
    'Needs Improvement': 0,
    'Average Performer': 1,
    'High Performer': 2
}
INVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

FEATURE_NAMES = [
    'exam_score',
    'attendance_percentage',
    'class_participation',
    'weekly_self_study_hours',
    'study_hours_per_day',
    'sleep_hours',
    'social_media_hours',
    'netflix_hours',
    'digital_distraction_hours',
    'study_to_distraction_ratio',
    'academic_composite',
    'academic_consistency',
    'exercise_frequency',
    'mental_health_rating'
]

def map_grade_to_performance(grade):
    if grade == 'A':
        return 'High Performer'
    elif grade == 'B':
        return 'Average Performer'
    else:  # C, D, F
        return 'Needs Improvement'

class StudentDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer fitted strictly on training data
    to eliminate data leakage.
    """
    def __init__(self):
        self.imputer = None
        self.scaler = None
        self.feature_names_out_ = FEATURE_NAMES

    def _engineer_features(self, df):
        d = df.copy()
        
        # Ensure base columns exist with defaults if missing
        if 'exam_score' not in d.columns:
            d['exam_score'] = d.get('mst_score', 70.0)
        if 'attendance_percentage' not in d.columns:
            d['attendance_percentage'] = d.get('attendance_rate', 75.0)
        if 'class_participation' not in d.columns:
            # If quiz_avg is supplied, map 0-100 down to 0-10
            d['class_participation'] = d.get('quiz_avg', 70.0) / 10.0
        if 'weekly_self_study_hours' not in d.columns:
            d['weekly_self_study_hours'] = d.get('study_hours_weekly', 15.0)
        if 'study_hours_per_day' not in d.columns:
            d['study_hours_per_day'] = d['weekly_self_study_hours'] / 7.0
        if 'sleep_hours' not in d.columns:
            d['sleep_hours'] = 7.0
        if 'social_media_hours' not in d.columns:
            d['social_media_hours'] = 2.5
        if 'netflix_hours' not in d.columns:
            d['netflix_hours'] = 1.5
        if 'exercise_frequency' not in d.columns:
            d['exercise_frequency'] = 3
        if 'mental_health_rating' not in d.columns:
            d['mental_health_rating'] = 7

        # 1. Digital distraction
        d['digital_distraction_hours'] = d['social_media_hours'] + d['netflix_hours']
        
        # 2. Ratio of study investment to entertainment
        d['study_to_distraction_ratio'] = d['weekly_self_study_hours'] / (d['digital_distraction_hours'] + 1.0)
        
        # 3. Scaled continuous assessment (0 - 100)
        participation_scaled = d['class_participation'] * 10.0
        
        # 4. Academic Composite index
        d['academic_composite'] = (
            0.40 * d['exam_score'] +
            0.30 * participation_scaled +
            0.30 * d['attendance_percentage']
        )
        
        # 5. Academic Consistency (Inverse of spread)
        pillars = np.column_stack([d['exam_score'], participation_scaled, d['attendance_percentage']])
        d['academic_consistency'] = 100.0 - np.std(pillars, axis=1)
        
        return d[FEATURE_NAMES]

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        X_eng = self._engineer_features(X_df)
        
        self.imputer = SimpleImputer(strategy='median')
        X_imputed = self.imputer.fit_transform(X_eng)
        
        self.scaler = StandardScaler()
        self.scaler.fit(X_imputed)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        X_eng = self._engineer_features(X_df)
        
        X_imputed = self.imputer.transform(X_eng)
        X_scaled = self.scaler.transform(X_imputed)
        return pd.DataFrame(X_scaled, columns=FEATURE_NAMES, index=X_df.index)

def load_excel_dataset(filepath="student_habits_performance.csv.xlsx"):
    df = pd.read_excel(filepath)
    
    # Drop empty columns from Excel (e.g., Unnamed: 16, 17, 18)
    unnamed_cols = [c for c in df.columns if 'Unnamed' in c]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        
    # Map letter grade to Pearson VUE target
    df['performance_category'] = df['grade'].apply(map_grade_to_performance)
    return df
