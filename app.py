"""
Flask Web Application for Pearson VUE Student Performance AI Classifier.
Trained on: student_habits_performance.csv.xlsx (UCS321: AI Fundamentals for Engineers)
"""

import os
import io
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import joblib

from src.preprocessing import LABEL_MAPPING, INVERSE_LABEL_MAPPING, FEATURE_NAMES
from src.predict import get_intervention_recommendation

app = Flask(__name__, template_folder='templates', static_folder='static')

PIPELINE_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model_pipeline.joblib")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "reports", "model_comparison.json")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "reports", "figures")

pipeline = None
if os.path.exists(PIPELINE_PATH):
    pipeline = joblib.load(PIPELINE_PATH)
    print("Inference pipeline loaded successfully.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        exam = float(data.get('mst_score', data.get('exam_score', 75.0)))
        attendance = float(data.get('attendance_rate', data.get('attendance_percentage', 80.0)))
        
        # class_participation: handle either 0-10 or 0-100% quiz
        quiz_raw = float(data.get('quiz_avg', data.get('class_participation', 7.0)))
        participation = quiz_raw / 10.0 if quiz_raw > 10.0 else quiz_raw
        
        study_hours = float(data.get('study_hours_weekly', data.get('weekly_self_study_hours', 15.0)))
        sleep = float(data.get('sleep_hours', 7.0))
        social_media = float(data.get('social_media_hours', 2.0))
        netflix = float(data.get('netflix_hours', 1.5))
        
        input_df = pd.DataFrame([{
            'exam_score': exam,
            'attendance_percentage': attendance,
            'class_participation': participation,
            'weekly_self_study_hours': study_hours,
            'study_hours_per_day': study_hours / 7.0,
            'sleep_hours': sleep,
            'social_media_hours': social_media,
            'netflix_hours': netflix,
            'exercise_frequency': 3,
            'mental_health_rating': 7
        }])

        pred_code = int(pipeline.predict(input_df)[0])
        pred_probs = pipeline.predict_proba(input_df)[0]
        category = INVERSE_LABEL_MAPPING[pred_code]

        screen_time = social_media + netflix
        result = get_intervention_recommendation(category, pred_probs, exam, attendance, study_hours, screen_time)
        
        # Calculate derived metrics
        composite = round(0.40 * exam + 0.30 * (participation * 10.0) + 0.30 * attendance, 2)
        distraction_ratio = round(study_hours / (screen_time + 1.0), 2)
        
        result['derived_metrics'] = {
            'composite_score': composite,
            'engagement_index': distraction_ratio,
            'study_hours_weekly': study_hours
        }

        return jsonify({'status': 'success', 'data': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file part provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No selected file'}), 400

        if file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)
            
        unnamed = [c for c in df.columns if 'Unnamed' in c]
        if unnamed:
            df = df.drop(columns=unnamed)
            
        pred_codes = pipeline.predict(df)
        pred_probs = pipeline.predict_proba(df)

        df['predicted_category'] = [INVERSE_LABEL_MAPPING[c] for c in pred_codes]
        df['confidence_score'] = [round(float(np.max(p)) * 100, 2) for p in pred_probs]
        
        for i, cat_name in INVERSE_LABEL_MAPPING.items():
            df[f'prob_{cat_name.lower().replace(" ", "_")}'] = [round(float(p[i]) * 100, 2) for p in pred_probs]

        dist = df['predicted_category'].value_counts().to_dict()
        total = len(df)
        dist_pct = {k: round((v / total) * 100, 1) for k, v in dist.items()}

        preview_data = df.head(15).fillna('N/A').to_dict(orient='records')

        output_buffer = io.StringIO()
        df.to_csv(output_buffer, index=False)
        output_csv_str = output_buffer.getvalue()

        return jsonify({
            'status': 'success',
            'summary': {
                'total_students': total,
                'distribution_counts': dist,
                'distribution_percentages': dist_pct,
                'avg_confidence': round(float(df['confidence_score'].mean()), 2)
            },
            'preview': preview_data,
            'csv_content': output_csv_str
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/model-metrics', methods=['GET'])
def model_metrics():
    try:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r') as f:
                data = json.load(f)
            return jsonify({'status': 'success', 'data': data})
        else:
            return jsonify({'status': 'error', 'message': 'Metrics file not found'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/reports/figures/<path:filename>')
def serve_figure(filename):
    return send_from_directory(FIGURES_DIR, filename)

@app.route('/api/download-sample-csv')
def download_sample_csv():
    excel_path = os.path.join(os.path.dirname(__file__), "student_habits_performance.csv.xlsx")
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path).head(30)
        unnamed = [c for c in df.columns if 'Unnamed' in c]
        if unnamed:
            df = df.drop(columns=unnamed)
        
        output_buffer = io.BytesIO()
        df.to_csv(output_buffer, index=False)
        output_buffer.seek(0)
        return send_file(
            output_buffer,
            as_attachment=True,
            download_name="sample_student_habits.csv",
            mimetype="text/csv"
        )
    return jsonify({'status': 'error', 'message': 'Dataset not found'}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"Starting Pearson VUE Classification Server on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
