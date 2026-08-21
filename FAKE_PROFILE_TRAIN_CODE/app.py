#!/usr/bin/env python3
"""
Flask web application for fake/real social media account detection.
Provides a web interface to input account features and get predictions.
"""

from flask import Flask, request, render_template_string
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and label encoder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'trained_model.pkl')
encoder_path = os.path.join(BASE_DIR, 'account_encoder.pkl')

try:
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    print("Model and encoder loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    label_encoder = None

# HTML template for the main page
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fake Account Detector</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #34495e;
        }
        input[type="number"], input[type="text"], select {
            width: 100%;
            padding: 10px;
            border: 1px solid #bdc3c7;
            border-radius: 4px;
            font-size: 16px;
            box-sizing: border-box;
        }
        input[type="number"]:focus, input[type="text"]:focus, select:focus {
            outline: none;
            border-color: #3498db;
            box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
        }
        .btn {
            background-color: #3498db;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 4px;
            font-size: 16px;
            cursor: pointer;
            width: 100%;
            margin-top: 20px;
        }
        .btn:hover {
            background-color: #2980b9;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            font-weight: bold;
            font-size: 18px;
        }
        .real {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .fake {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .features-info {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 25px;
            font-size: 14px;
            color: #6c757d;
        }
        .feature-list {
            columns: 2;
            -webkit-columns: 2;
            -moz-columns: 2;
        }
        .feature-list li {
            margin-bottom: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Fake Account Detector</h1>

        <div class="features-info">
            <strong>Enter account features to predict if it's Fake or Real:</strong>
            <ul class="feature-list">
                <li>Followers: Number of followers</li>
                <li>Following: Number of accounts followed</li>
                <li>Posts: Number of posts</li>
                <li>Engagement Rate (%): Percentage of engagement</li>
                <li>Avg Likes per Post: Average likes per post</li>
                <li>Avg Comments per Post: Average comments per post</li>
                <li>Verified: 0 (No) or 1 (Yes)</li>
                <li>Account Age (Years): Age of account in years</li>
                <li>Bio Text: Biography text description</li>
            </ul>
        </div>

        <form method="POST" action="/predict">
            <div class="form-group">
                <label for="followers">Followers:</label>
                <input type="number" id="followers" name="Followers" min="0" required>
            </div>

            <div class="form-group">
                <label for="following">Following:</label>
                <input type="number" id="following" name="Following" min="0" required>
            </div>

            <div class="form-group">
                <label for="posts">Posts:</label>
                <input type="number" id="posts" name="Posts" min="0" required>
            </div>

            <div class="form-group">
                <label for="engagement">Engagement Rate (%):</label>
                <input type="number" id="engagement" name="Engagement Rate (%)" min="0" step="0.01" required>
            </div>

            <div class="form-group">
                <label for="avg_likes">Avg Likes per Post:</label>
                <input type="number" id="avg_likes" name="Avg Likes per Post" min="0" required>
            </div>

            <div class="form-group">
                <label for="avg_comments">Avg Comments per Post:</label>
                <input type="number" id="avg_comments" name="Avg Comments per Post" min="0" required>
            </div>

            <div class="form-group">
                <label for="verified">Verified:</label>
                <select id="verified" name="Verified" required>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
            </div>

            <div class="form-group">
                <label for="age">Account Age (Years):</label>
                <input type="number" id="age" name="Account Age (Years)" min="0" step="0.1" required>
            </div>

            <div class="form-group">
                <label for="bio">Bio Text:</label>
                <input type="text" id="bio" name="Bio Text" placeholder="Enter bio description..." required>
            </div>

            <button type="submit" class="btn">Predict Account Type</button>
        </form>

        {% if prediction %}
        <div class="result {{ prediction|lower }}">
            Prediction: <strong>{{ prediction }}</strong>
        </div>
        {% endif %}

        {% if error %}
        <div class="result error">
            Error: {{ error }}
        </div>
        {% endif %}

        <div style="margin-top: 30px; text-align: center; color: #7f8c8d; font-size: 14px;">
            <p>Powered by Machine Learning Model</p>
        </div>
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET'])
def index():
    """Render the main page with the input form."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission and make prediction."""
    if model is None or label_encoder is None:
        return render_template_string(HTML_TEMPLATE,
                                    error="Model not loaded. Please check server logs.")

    try:
        # Get form data
        input_data = {
            'Followers': float(request.form['Followers']),
            'Following': float(request.form['Following']),
            'Posts': float(request.form['Posts']),
            'Engagement Rate (%)': float(request.form['Engagement Rate (%)']),
            'Avg Likes per Post': float(request.form['Avg Likes per Post']),
            'Avg Comments per Post': float(request.form['Avg Comments per Post']),
            'Verified': float(request.form['Verified']),
            'Account Age (Years)': float(request.form['Account Age (Years)']),
            'Bio Text': request.form['Bio Text']
        }

        # Make prediction
        import pandas as pd
        df_input = pd.DataFrame([input_data])
        prediction_encoded = model.predict(df_input)[0]
        prediction = label_encoder.inverse_transform([prediction_encoded])[0]

        # Return result
        return render_template_string(HTML_TEMPLATE, prediction=prediction)

    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error=f"Prediction error: {str(e)}")

if __name__ == '__main__':
    print("Starting Fake Account Detector web application...")
    print("Access the application at: http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)