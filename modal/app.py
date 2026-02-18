from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
from pydantic import BaseModel, Field, ValidationError
from typing import Optional
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ----------------------------
# CORS HEADERS
# ----------------------------
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# ----------------------------
# GLOBAL VARIABLES
# ----------------------------
model = None
scaler = None
feature_cols = None
label_encoders = {}

DATASET_PATH = "CubanDataset.csv"
USER_DATA_PATH = "user_submissions.csv"

# ----------------------------
# PYDANTIC MODEL
# ----------------------------
class BreastCancerInput(BaseModel):
    age: float = Field(..., ge=0, le=120)
    menarche: float = Field(..., ge=0, le=30)
    menopause: float = Field(..., ge=0, le=80)
    agefirst: float = Field(..., ge=0, le=60)
    children: float = Field(..., ge=0, le=20)
    breastfeeding: float = Field(..., ge=0, le=600)
    nrelbc: str
    biopsies: float = Field(..., ge=0)
    hyperplasia: str
    race: str
    year: float
    imc: float
    weight: float
    exercise: str
    alcohol: str
    tobacco: str
    allergies: str
    emotional: str
    depressive: str
    histologicalclass: float
    birads: float

# ----------------------------
# SAVE USER DATA
# ----------------------------
def save_user_data(validated_data, prediction=None):
    try:
        data_dict = validated_data.dict()
        data_dict["prediction"] = prediction
        data_dict["timestamp"] = pd.Timestamp.now()

        df_new = pd.DataFrame([data_dict])

        if not os.path.exists(USER_DATA_PATH):
            df_new.to_csv(USER_DATA_PATH, index=False)
        else:
            df_new.to_csv(USER_DATA_PATH, mode='a', header=False, index=False)

        print("User data saved successfully")
    except Exception as e:
        print("Error saving user data:", e)

# ----------------------------
# SAFE FLOAT CONVERSION
# ----------------------------
def safe_float_conversion(value, default=0):
    try:
        return float(value)
    except:
        return default

# ----------------------------
# TRAIN MODEL
# ----------------------------
def train_model():
    global model, scaler, feature_cols, label_encoders

    try:
        df = pd.read_csv(DATASET_PATH)

        df['cancer'] = df['cancer'].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})

        feature_cols = [
            'age','menarche','menopause','agefirst','children',
            'breastfeeding','nrelbc','biopsies','hyperplasia',
            'race','year','imc','weight','exercise','alcohol',
            'tobacco','allergies','emotional','depressive',
            'histologicalclass','birads'
        ]

        # Encode categorical columns
        categorical_cols = [
            'nrelbc','race','exercise','alcohol',
            'tobacco','allergies','emotional',
            'depressive','hyperplasia'
        ]

        for col in categorical_cols:
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        # Fill missing numeric values
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())

        X = df[feature_cols]
        y = df['cancer']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )

        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Model trained successfully. Accuracy: {accuracy:.4f}")
        return True

    except Exception as e:
        print("Training error:", e)
        return False

# ----------------------------
# ROUTES
# ----------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        raw_json = request.get_json()

        # Validate input
        try:
            validated_input = BreastCancerInput(**raw_json)
        except ValidationError as e:
            return jsonify({'success': False, 'errors': e.errors()})

        # Prepare model input
        input_data = []

        for col in feature_cols:
            value = getattr(validated_input, col)

            if col in label_encoders:
                le = label_encoders[col]
                if value in le.classes_:
                    encoded = le.transform([value])[0]
                else:
                    encoded = 0
                input_data.append(encoded)
            else:
                input_data.append(safe_float_conversion(value))

        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        probability = model.predict_proba(input_scaled)[0, 1]
        current_risk = float(probability * 100)
        three_year_risk = float(min(99, current_risk * 1.3))

        if current_risk > 30:
            risk_level = "HIGH"
            recommendation = "Consult healthcare professional immediately"
        elif current_risk > 15:
            risk_level = "MODERATE"
            recommendation = "Regular screening recommended"
        else:
            risk_level = "LOW"
            recommendation = "Continue regular checkups"

        # Save validated user data
        save_user_data(validated_input, current_risk)

        return jsonify({
            'success': True,
            'current_risk': round(current_risk, 2),
            'three_year_risk': round(three_year_risk, 2),
            'risk_level': risk_level,
            'recommendation': recommendation
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/model_info')
def model_info():
    return jsonify({
        'success': True,
        'features': feature_cols
    })

@app.route('/test')
def test():
    return jsonify({'status': 'Server running'})

# ----------------------------
# RUN SERVER
# ----------------------------
if __name__ == '__main__':
    print("Training model...")
    if train_model():
        print("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Model training failed.")