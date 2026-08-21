import pandas as pd
import numpy as np
import random

# Load original data
df = pd.read_csv('dataset_original.csv')
print(f"Original shape: {df.shape}")

# Identify column types
numeric_cols = ['Followers', 'Following', 'Posts', 'Engagement Rate (%)',
                'Avg Likes per Post', 'Avg Comments per Post', 'Account Age (Years)']
binary_cols = ['Verified']
categorical_cols = ['Account Type']  # also the target
text_col = 'Bio Text'

# Compute stats for numeric columns
numeric_stats = {}
for col in numeric_cols:
    numeric_stats[col] = {
        'min': df[col].min(),
        'max': df[col].max(),
        'mean': df[col].mean(),
        'std': df[col].std()
    }

# Unique values for categorical
account_types = df['Account Type'].unique().tolist()
verified_vals = df['Verified'].unique().tolist()
bios = df['Bio Text'].tolist()

# Number of synthetic rows to generate
n_synthetic = 5000  # adjust as needed
synthetic_rows = []

for _ in range(n_synthetic):
    row = {}
    # numeric: sample from normal distribution clipped to min/max
    for col in numeric_cols:
        mean = numeric_stats[col]['mean']
        std = numeric_stats[col]['std']
        val = np.random.normal(mean, std)
        # clip
        val = max(numeric_stats[col]['min'], min(val, numeric_stats[col]['max']))
        # if column is integer-like, round
        if col in ['Followers', 'Following', 'Posts', 'Avg Likes per Post', 'Avg Comments per Post', 'Account Age (Years)']:
            val = int(round(val))
        else:
            # keep as float for Engagement Rate
            pass
        row[col] = val
    # binary
    row['Verified'] = np.random.choice(verified_vals)
    # categorical
    row['Account Type'] = np.random.choice(account_types)
    # text
    row['Bio Text'] = np.random.choice(bios)
    synthetic_rows.append(row)

synthetic_df = pd.DataFrame(synthetic_rows)
# Ensure column order same as original
synthetic_df = synthetic_df[df.columns]
# Combine
df_combined = pd.concat([df, synthetic_df], ignore_index=True)
print(f"Combined shape: {df_combined.shape}")
# Save
df_combined.to_csv('dataset.csv', index=False)
print("Saved augmented dataset to dataset.csv")