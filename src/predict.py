"""
UCS321: Inference Engine for Student Performance Classification
Trained on: student_habits_performance.csv.xlsx

Supports:
1. Single student CLI inference with probability breakdown & intervention advice
2. Batch CSV prediction
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import LABEL_MAPPING, INVERSE_LABEL_MAPPING

def get_intervention_recommendation(category, probs, exam, attendance, study_hours, screen_time):
    """
    Actionable early warning educational intervention plan.
    """
    if category == 'Needs Improvement':
        urgency = "HIGH RISK - IMMEDIATE INTERVENTION REQUIRED"
        actions = [
            f"Mandatory attendance counseling (current: {attendance:.1f}%)." if attendance < 75 else "Maintain consistent lecture presence.",
            f"Remedial tutoring on foundational exam topics (Exam score: {exam:.1f}).",
            f"Digital distraction management: Reduce recreational screen time (currently {screen_time:.1f} hrs/day).",
            f"Target minimum self-study threshold of 15 hrs/week (currently {study_hours:.1f} hrs/week).",
            "Pair with a High Performer student peer tutor."
        ]
    elif category == 'Average Performer':
        urgency = "MODERATE RISK - PROGRESSION OPPORTUNITIES"
        actions = [
            f"Strengthen continuous assessment and quiz participation (Exam score: {exam:.1f}).",
            f"Convert average assignments into distinctions by attending weekly faculty office hours.",
            f"Optimize balance between study ({study_hours:.1f} hrs/wk) and screen time ({screen_time:.1f} hrs/day).",
            "Participate in weekly problem-solving circles."
        ]
    else:  # High Performer
        urgency = "OPTIMAL TRAJECTORY - ADVANCED MASTERY & ENRICHMENT"
        actions = [
            "Encourage participation in advanced honors tracks and research projects.",
            "Nominate for student peer tutor / laboratory assistant fellowship.",
            "Prepare for Pearson VUE industry certification testing.",
            "Maintain healthy sleep habits and avoid academic burnout."
        ]
        
    return {
        "urgency_level": urgency,
        "recommendations": actions,
        "predicted_category": category,
        "confidence": round(float(np.max(probs)) * 100, 2),
        "class_probabilities": {
            INVERSE_LABEL_MAPPING[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(probs))
        }
    }

def predict_single(pipeline_path, exam, attendance, participation, study_hours, sleep=7.0, social_media=2.0, netflix=1.5):
    pipeline = joblib.load(pipeline_path)
    
    input_df = pd.DataFrame([{
        'exam_score': float(exam),
        'attendance_percentage': float(attendance),
        'class_participation': float(participation),
        'weekly_self_study_hours': float(study_hours),
        'study_hours_per_day': float(study_hours) / 7.0,
        'sleep_hours': float(sleep),
        'social_media_hours': float(social_media),
        'netflix_hours': float(netflix),
        'exercise_frequency': 3,
        'mental_health_rating': 7
    }])
    
    pred_code = int(pipeline.predict(input_df)[0])
    pred_probs = pipeline.predict_proba(input_df)[0]
    category = INVERSE_LABEL_MAPPING[pred_code]
    
    screen_time = float(social_media) + float(netflix)
    return get_intervention_recommendation(category, pred_probs, exam, attendance, study_hours, screen_time)

def predict_batch(pipeline_path, input_file, output_csv):
    pipeline = joblib.load(pipeline_path)
    
    if input_file.endswith('.xlsx') or input_file.endswith('.xls'):
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)
        
    pred_codes = pipeline.predict(df)
    pred_probs = pipeline.predict_proba(df)
    
    results_df = df.copy()
    results_df['predicted_category'] = [INVERSE_LABEL_MAPPING[c] for c in pred_codes]
    results_df['confidence_score'] = [round(float(np.max(p)) * 100, 2) for p in pred_probs]
    
    for i, cat_name in INVERSE_LABEL_MAPPING.items():
        results_df[f'prob_{cat_name.lower().replace(" ", "_")}'] = [round(float(p[i]) * 100, 2) for p in pred_probs]
        
    results_df.to_csv(output_csv, index=False)
    print(f"Batch prediction complete! {len(df)} records processed.")
    print(f"Results saved to: {output_csv}")
    print("\nSummary distribution:")
    print(results_df['predicted_category'].value_counts())
    return results_df

def main():
    parser = argparse.ArgumentParser(description="UCS321 Student Performance Inference Engine")
    parser.add_argument("--model", default="models/best_model_pipeline.joblib", help="Path to pipeline model")
    parser.add_argument("--batch", type=str, help="Path to input file for batch scoring")
    parser.add_argument("--output", default="reports/batch_predictions.csv", help="Path for output")
    
    parser.add_argument("--exam", type=float, default=75.0, help="Exam / MST score (0-100)")
    parser.add_argument("--attendance", type=float, default=85.0, help="Attendance percentage (0-100)")
    parser.add_argument("--participation", type=float, default=7.0, help="Class participation / quiz (0-10)")
    parser.add_argument("--study-hours", type=float, default=15.0, help="Weekly self study hours (0-40)")
    parser.add_argument("--sleep", type=float, default=7.0, help="Daily sleep hours (0-12)")
    parser.add_argument("--social-media", type=float, default=2.0, help="Daily social media hours (0-12)")
    parser.add_argument("--netflix", type=float, default=1.5, help="Daily netflix streaming hours (0-12)")
    
    args = parser.parse_args()
    
    if args.batch:
        predict_batch(args.model, args.batch, args.output)
    else:
        res = predict_single(
            args.model, args.exam, args.attendance, args.participation,
            args.study_hours, args.sleep, args.social_media, args.netflix
        )
        print("\n" + "=" * 65)
        print("PEARSON VUE STUDENT EARLY WARNING ASSESSMENT")
        print("=" * 65)
        print(f"Predicted Category: {res['predicted_category'].upper()}")
        print(f"Confidence Level:   {res['confidence']}%")
        print("\nClass Probabilities:")
        for cat, prob in res['class_probabilities'].items():
            bar = '█' * int(prob / 4)
            print(f"  • {cat:<18}: {prob:>5.1f}% |{bar:<25}|")
        print(f"\nStatus: {res['urgency_level']}")
        print("\nTailored Action Plan:")
        for rec in res['recommendations']:
            print(f"  ✓ {rec}")
        print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
