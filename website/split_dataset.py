import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('dataset.csv')
print(f'Dataset shape: {df.shape}')
# Ensure target column exists
if 'Account Type' not in df.columns:
    raise ValueError("Account Type column not found")
# Separate features and target
X = df.drop(columns=['Account Type'])
y = df['Account Type']
# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Combine back
train_df = X_train.copy()
train_df['Account Type'] = y_train
test_df = X_test.copy()
test_df['Account Type'] = y_test
print(f'Train shape: {train_df.shape}')
print(f'Test shape: {test_df.shape}')
# Save
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
print('Saved train.csv and test.csv')