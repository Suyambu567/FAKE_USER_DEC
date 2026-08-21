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

# Save model and label encoder for target
joblib.dump(model, 'trained_model.pkl')
le_account = LabelEncoder()
le_account.fit(y)  # fit on all labels
joblib.dump(le_account, 'account_encoder.pkl')
print("Model and encoder saved.")

# Load to verify
loaded_model = joblib.load('trained_model.pkl')
loaded_le = joblib.load('account_encoder.pkl')
print("Model and encoder loaded.")

# Example prediction function (only runs when script is executed directly)
def make_prediction(input_data):
    # input_data dict with same keys as X
    df_input = pd.DataFrame([input_data])
    pred = loaded_model.predict(df_input)
    return pred

if __name__ == "__main__":
    # Example input
    new_data = {
        'Followers': 5000,
        'Following': 300,
        'Posts': 150,
        'Engagement Rate (%)': 4.5,
        'Avg Likes per Post': 400,
        'Avg Comments per Post': 20,
        'Verified': 0,
        'Account Age (Years)': 5,
        'Bio Text': 'Foodie | Reviews and recipes'
    }

    # Make a prediction
    prediction = make_prediction(new_data)
    print(f"\nPredicted Account Type: {prediction[0]}")

    # Evaluate on test set
    y_pred = loaded_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=loaded_le.classes_))