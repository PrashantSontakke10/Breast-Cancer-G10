from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Add CORS headers manually
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# ... (keep the rest of your existing app.py code the same) ...

# Global variables for model and scaler
model = None
scaler = None
feature_cols = None
label_encoders = {}

def extract_breastfeeding_months(x):
    if pd.isna(x) or x in ['No', 'no', 'None', '', '0']:
        return 0
    elif isinstance(x, str):
        if 'month' in x.lower():
            try:
                return float(''.join(filter(str.isdigit, x)) or 0)
            except:
                return 0
        elif 'year' in x.lower():
            try:
                years = float(''.join(filter(str.isdigit, x)) or 0)
                return years * 12
            except:
                return 0
    try:
        return float(x)
    except:
        return 0

def convert_menopause(x):
    if pd.isna(x) or x in ['No', 'no', 'None', '']:
        return 0
    try:
        return float(x)
    except:
        return 0

def safe_float_conversion(value, default=0):
    """Safely convert value to float, return default if conversion fails"""
    if value is None or value == '':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def train_model():
    global model, scaler, feature_cols, label_encoders
    
    try:
        # Load dataset
        df = pd.read_csv('CubanDataset.csv')
        print("Dataset loaded successfully")
        
        # Preprocessing
        df['cancer'] = df['cancer'].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
        
        expected_columns = ['age', 'menarche', 'menopause', 'agefirst', 'children', 
                          'breastfeeding', 'nrelbc', 'biopsies', 'hyperplasia', 
                          'race', 'year', 'imc', 'weight', 'exercise', 'alcohol', 
                          'tobacco', 'allergies', 'emotional', 'depressive', 
                          'histologicalclass', 'birads']
        
        for col in expected_columns:
            if col not in df.columns:
                if col in ['age', 'menarche', 'agefirst', 'children', 'biopsies', 'year', 'imc', 'weight', 'histologicalclass', 'birads']:
                    df[col] = 0
                else:
                    df[col] = 'Unknown'
        
        df['menopause'] = df['menopause'].apply(convert_menopause)
        df['breastfeeding'] = df['breastfeeding'].apply(extract_breastfeeding_months)
        
        categorical_cols = ['nrelbc', 'race', 'exercise', 'alcohol', 'tobacco', 
                           'allergies', 'emotional', 'depressive', 'hyperplasia']
        
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('Unknown')
                le = LabelEncoder()
                le.fit(list(df[col].unique()) + ['Unknown', 'No', 'Yes'])
                df[col] = le.transform(df[col])
                label_encoders[col] = le
        
        numerical_cols = ['age', 'menarche', 'menopause', 'agefirst', 'children', 
                         'breastfeeding', 'biopsies', 'year', 'imc', 'weight', 
                         'histologicalclass', 'birads']
        
        for col in numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())
        
        feature_cols = [col for col in expected_columns if col in df.columns]
        X = df[feature_cols].copy()
        y = df['cancer']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
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
        
        print(f"Model trained successfully with accuracy: {accuracy:.4f}")
        print(f"Features used: {feature_cols}")
        return True
        
    except Exception as e:
        print(f"Error training model: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)  # Debug print
        
        # Prepare input data with safe conversion
        input_data = []
        for col in feature_cols:
            value = data.get(col, '0')  # Default to '0' if missing
            
            # Handle different data types
            if col in ['nrelbc', 'race', 'exercise', 'alcohol', 'tobacco', 
                      'allergies', 'emotional', 'depressive', 'hyperplasia']:
                # Categorical features
                input_data.append(safe_float_conversion(value, 0))
            else:
                # Numerical features
                input_data.append(safe_float_conversion(value, 0))
        
        print("Processed input data:", input_data)  # Debug print
        
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        probability = model.predict_proba(input_scaled)[0, 1]
        # Convert numpy float32 to Python float for JSON serialization
        current_risk = float(probability * 100)
        three_year_risk = float(min(99, current_risk * 1.3))
        
        # Determine risk level
        if current_risk > 30:
            risk_level = "HIGH"
            recommendation = "Please consult a healthcare professional immediately"
        elif current_risk > 15:
            risk_level = "MODERATE"
            recommendation = "Regular screening recommended"
        else:
            risk_level = "LOW"
            recommendation = "Continue with regular checkups"
        
        return jsonify({
            'success': True,
            'current_risk': round(current_risk, 2),
            'three_year_risk': round(three_year_risk, 2),
            'risk_level': risk_level,
            'recommendation': recommendation
        })
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/model_info')
def model_info():
    if model is None:
        return jsonify({'success': False, 'error': 'Model not trained'})
    
    return jsonify({
        'success': True,
        'features': feature_cols,
        'accuracy': 'Trained successfully'
    })

if __name__ == '__main__':
    print("Training model...")
    if train_model():
        print("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to train model. Please check your dataset.")
        
@app.route('/test', methods=['GET'])
def test_endpoint():
    return jsonify({'message': 'Server is working!', 'status': 'ok'})

@app.route('/test-predict', methods=['POST'])
def test_predict():
    try:
        # Test with sample data
        test_data = [45, 12, 0, 25, 2, 6, 0, 0, 0, 0, 2024, 25, 65, 2, 0, 0, 0, 0, 0, 3, 3]
        input_array = np.array(test_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        probability = model.predict_proba(input_scaled)[0, 1]
        
        return jsonify({
            'success': True,
            'current_risk': round(float(probability * 100), 2),
            'test': 'manual_test'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})