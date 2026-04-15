from flask import Flask, render_template, request, jsonify
from ml_model import predict
from health_service import calculate_bmi
from helpers import hash_password
from db import get_connection

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def analyze_health():
    try:
        data = request.json
        symptom = data.get('symptom', '')

        if not symptom:
            return jsonify({'error': 'Symptom required'}), 400

        result = predict(symptom)

        return jsonify({
            'symptom': symptom,
            'recommendation': result,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bmi', methods=['POST'])
def calculate_bmi_api():
    try:
        data = request.json
        weight = float(data.get('weight'))
        height = float(data.get('height'))

        bmi = calculate_bmi(weight, height)

        return jsonify({
            'weight': weight,
            'height': height,
            'bmi': round(bmi, 2),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# IMPORTANT for Render
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
