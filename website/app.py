from flask import Flask, request, render_template, flash, redirect, url_for
import joblib
import pandas as pd
import os
import sys
import json
import numpy as np

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'  # Needed for flashing messages

# Load the trained model and encoders with error handling
def load_model():
    try:
        model_path = 'trained_model.pkl'
        bio_encoder_path = 'bio_encoder.pkl'
        account_encoder_path = 'account_encoder.pkl'

        if not all(os.path.exists(p) for p in [model_path, bio_encoder_path, account_encoder_path]):
            raise FileNotFoundError('Model files not found. Please train the model first.')

        loaded_model = joblib.load(model_path)
        le_bio = joblib.load(bio_encoder_path)
        le_account = joblib.load(account_encoder_path)
        return loaded_model, le_bio, le_account
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return None, None, None

loaded_model, le_bio, le_account = load_model()

if loaded_model is None:
    # In a production app, you might want to raise an exception or handle this differently
    print("WARNING: Model not loaded. Some functionality will be unavailable.", file=sys.stderr)

# Prediction function with input validation
def make_prediction(input_data):
    """
    Make a prediction using the loaded model and encoders.
    Returns (prediction_label, confidence_score)
    """
    try:
        if loaded_model is None or le_bio is None or le_account is None:
            raise Exception("Model not loaded. Please train the model first.")

        # Transform the Bio Text using the loaded encoder
        if input_data['Bio Text'].strip() == '':
            raise ValueError("Bio Text cannot be empty")

        bio_text_encoded = le_bio.transform([input_data['Bio Text']])[0]
        input_data['Bio Text'] = bio_text_encoded

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Get prediction
        prediction = loaded_model.predict(input_df)[0]
        prediction_label = le_account.inverse_transform([prediction])[0]

        # Get prediction probabilities if available (for confidence)
        confidence = None
        if hasattr(loaded_model, "predict_proba"):
            probs = loaded_model.predict_proba(input_df)[0]
            # Assuming classes are ordered as le_account.classes_
            confidence = max(probs)

        return prediction_label, confidence
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/analytics')
def analytics():
    # Load model and data if not already loaded
    global loaded_model, le_bio, le_account
    if loaded_model is None:
        loaded_model, le_bio, le_account = load_model()

    # Initialize default values
    model_accuracy = "0.68"  # Placeholder
    avg_prediction_time = "0.8"  # Placeholder
    training_samples = "0"
    features_used = "0"
    feature_names_json = json.dumps(['Followers', 'Following', 'Posts', 'Engagement Rate (%)',
                                   'Avg Likes per Post', 'Avg Comments per Post', 'Verified',
                                   'Account Age (Years)', 'Bio Text'])
    importances_json = json.dumps([0.18, 0.12, 0.15, 0.22, 0.10, 0.08, 0.05, 0.05, 0.05])
    fake_count = 0
    real_count = 0
    prediction_percentages = ["0.0%", "0.0%"]
    prediction_percentages_json = json.dumps(["0.0%", "0.0%"])
    engagement_bins_json = json.dumps(['0-1%', '1-2%', '2-3%', '3-4%', '4-5%', '5-6%', '6-7%', '7-8%', '8-9%', '9-10%', '10-15%', '15-20%+'])
    engagement_counts_json = json.dumps([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # Try to load dataset and compute real statistics if possible
    if loaded_model is not None:
        try:
            df = pd.read_csv('dataset.csv')

            # Calculate statistics
            training_samples = len(df)
            features_used = len(df.columns) - 1  # Subtract Account Type column

            # Calculate fake/real distribution
            fake_count = len(df[df['Account Type'] == 'Fake'])
            real_count = len(df[df['Account Type'] == 'Real'])
            total = fake_count + real_count
            if total > 0:
                fake_percentage = round((fake_count / total) * 100, 1)
                real_percentage = round((real_count / total) * 100, 1)
                prediction_percentages = [f"{fake_percentage}%", f"{real_percentage}%"]
                prediction_percentages_json = json.dumps(prediction_percentages)

            # Engagement rate distribution (for line chart - histogram style)
            engagement_rates = df['Engagement Rate (%)'].values
            # Create bins for histogram
            bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
            labels = ['0-1%', '1-2%', '2-3%', '3-4%', '4-5%', '5-6%', '6-7%', '7-8%', '8-9%', '9-10%', '10-15%', '15-20%+']
            counts, _ = np.histogram(engagement_rates, bins=bins)
            engagement_bins_json = json.dumps(labels)
            engagement_counts_json = json.dumps(counts.tolist())

            # Feature names and importances (from the model)
            try:
                # Get feature names after ColumnTransformer transformation
                preprocessor = loaded_model.named_steps['preprocessor']
                classifier = loaded_model.named_steps['classifier']

                # Get feature names from the column transformer
                feature_names = []
                for name, trans, column in preprocessor.transformers_:
                    if trans == 'passthrough':
                        if isinstance(column, str):
                            feature_names.append(column)
                        else:
                            feature_names.extend(column)
                    elif hasattr(trans, 'get_feature_names_out'):
                        if isinstance(column, str):
                            feature_names.extend(trans.get_feature_names_out([column]))
                        else:
                            feature_names.extend(trans.get_feature_names_out(column))
                    else:
                        # Fallback
                        if isinstance(column, str):
                            feature_names.append(column)
                        else:
                            feature_names.extend(column)

                # Get feature importances from the classifier
                importances = classifier.feature_importances_

                # Format for JavaScript
                feature_names_json = json.dumps(feature_names[:len(importances)])
                importances_json = json.dumps(importances.tolist())
            except Exception as e:
                print(f"Could not extract feature names/importances: {e}", file=sys.stderr)
                # Fallback values
                feature_names = ['Followers', 'Following', 'Posts', 'Engagement Rate (%)',
                               'Avg Likes per Post', 'Avg Comments per Post', 'Verified',
                               'Account Age (Years)', 'Bio Text']
                importances = [0.18, 0.12, 0.15, 0.22, 0.10, 0.08, 0.05, 0.05, 0.05]
                feature_names_json = json.dumps(feature_names)
                importances_json = json.dumps(importances)

        except Exception as e:
            print(f"Could not load dataset for analytics: {e}", file=sys.stderr)
            # Keep default values

    return render_template('analytics.html',
                         model_accuracy=model_accuracy,
                         avg_prediction_time=avg_prediction_time,
                         training_samples=training_samples,
                         features_used=features_used,
                         feature_names_json=feature_names_json,
                         feature_importances=importances_json,
                         fake_count=fake_count,
                         real_count=real_count,
                         prediction_percentages=prediction_percentages_json,
                         engagement_bins=engagement_bins_json,
                         engagement_counts=engagement_counts_json)

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/word-analysis')
def word_analysis():
    return render_template('word_analysis.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Predict function called", file=sys.stdout)
    if loaded_model is None or le_bio is None or le_account is None:
        flash('Model not available. Please train the model first or contact administrator.', 'error')
        return redirect(url_for('home'))

    try:
        # Get and validate form data
        form_data = request.form
        print(f"Form data received: {dict(form_data)}", file=sys.stdout)

        # Define expected fields and their types
        fields = {
            'Followers': ('int', lambda x: int(x) >= 0),
            'Following': ('int', lambda x: int(x) >= 0),
            'Posts': ('int', lambda x: int(x) >= 0),
            'Engagement Rate (%)': ('float', lambda x: 0 <= float(x) <= 100),
            'Avg Likes per Post': ('int', lambda x: int(x) >= 0),
            'Avg Comments per Post': ('int', lambda x: int(x) >= 0),
            'Verified': ('int', lambda x: int(x) in [0, 1]),
            'Account Age (Years)': ('int', lambda x: int(x) >= 0),
            'Bio Text': ('str', lambda x: len(x.strip()) > 0)
        }

        input_data = {}
        errors = []

        for field, (type_check, validator) in fields.items():
            value = form_data.get(field, '').strip()
            print(f"Processing field '{field}': value='{value}'", file=sys.stdout)

            if not value:
                errors.append(f'{field} is required.')
                continue

            try:
                if type_check == 'int':
                    value = int(value)
                elif type_check == 'float':
                    value = float(value)
                # For str, keep as string

                if not validator(value):
                    errors.append(f'{field} has an invalid value.')
                else:
                    input_data[field] = value
            except ValueError:
                errors.append(f'{field} must be a valid {type_check}.')

        print(f"Validation errors: {errors}", file=sys.stdout)
        if errors:
            for error in errors:
                flash(error, 'error')
            return redirect(url_for('home'))

        # Make prediction
        predicted_account_type, confidence = make_prediction(input_data)

        # Prepare confidence percentage for display
        confidence_percent = f"{confidence*100:.1f}%" if confidence is not None else "N/A"

        return render_template('result.html',
                             prediction=predicted_account_type,
                             confidence=confidence_percent)

    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stdout)
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('home'))

if __name__ == '__main__':
    print("Starting Fake Account Detector web application...")
    print("Access the application at: http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)