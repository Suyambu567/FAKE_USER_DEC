import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import joblib
import os
import numpy as np
# Load the dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "dataset.csv")
df = pd.read_csv(dataset_path)
print("Dataset loaded. Shape:", df.shape)

# Separate features and target
X = df.drop('Account Type', axis=1)
y = df['Account Type']

# Identify text and numeric columns
text_col = 'Bio Text'
numeric_cols = [c for c in X.columns if c != text_col]

# Preprocessing for text: TF-IDF
text_transformer = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(max_features=100, stop_words='english'))
])

# Preprocessing for numeric: standard scaling
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('txt', text_transformer, text_col)
    ])

# Create pipeline with classifier
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train
model.fit(X_train, y_train)

# Save model and label encoder for target (if needed for inverse transform)
joblib.dump(model, 'trained_model_improved.pkl')
le_account = LabelEncoder()
le_account.fit(y)  # fit on all labels
joblib.dump(le_account, 'account_encoder_improved.pkl')
print("Improved model and encoder saved.")

# Load to verify
loaded_model = joblib.load('trained_model_improved.pkl')
loaded_le = joblib.load('account_encoder_improved.pkl')
print("Model and encoder loaded.")

# Predict
y_pred = loaded_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=loaded_le.classes_))

# Example prediction function
def make_prediction_improved(input_data):
    # input_data dict with same keys as X
    df_input = pd.DataFrame([input_data])
    pred = loaded_model.predict(df_input)
    return pred

# Example input
new_data = {
    'Followers': 4473,
    'Following': 1188,
    'Posts': 360,
    'Engagement Rate (%)': 5.12,
    'Avg Likes per Post': 911,
    'Avg Comments per Post': 54,
    'Verified': 0,
    'Account Age (Years)': 11,
    'Bio Text': 'Foodie | Reviews and recipes'
}
pred_improved = make_prediction_improved(new_data)
print(f"\nImproved model prediction: {pred_improved[0]}")

# Additional debugging
print('Prediction sample:', y_pred[:5])
print('Prediction type:', type(y_pred[0]))
print('Unique preds:', np.unique(y_pred))

try:
    inv = loaded_le.inverse_transform(y_pred)
    print('Inverse transform succeeded')
except Exception as e:
    print('Error:', e)
    print('LE classes:', loaded_le.classes_)
    print('Pred unique:', np.unique(y_pred))