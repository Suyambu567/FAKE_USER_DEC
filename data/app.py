import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Number of samples
num_samples = 10000

# Sample names
first_names = ['Emma', 'Olivia', 'Ava', 'Isabella', 'Sophia', 'Mia', 'Charlotte', 'Amelia', 'Evelyn', 'Abigail']
last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']

# Sample bio texts
bio_texts = [
    'Passionate traveler and food lover', 
    'Love exploring new places 🌎',
    'Photographer and content creator 📸',
    'Entrepreneur & lifestyle blogger',
    'Digital nomad living my best life 🌴',
    'Sharing moments from my journey',
    'Personal blog | Stay positive ✨',
    'Fitness enthusiast 💪 | Health advocate',
    'DM for collaborations or inquiries!',
    'Nature lover | Saving the planet 🌍',
    'Aspiring artist | Drawing my dreams',
    'Business inquiries? Contact below 👇',
    'Foodie | Reviews and recipes',
    'Check out my website for more updates!',
    'Follow for daily inspiration 🌟'
]

# Random date generation for account creation (last 10 years)
def random_date(start_year=2013):
    start_date = datetime(start_year, 1, 1)
    random_days = np.random.randint(0, 365*10)  # up to 10 years back
    return start_date + timedelta(days=random_days)

# Generate synthetic data
data = {
    'Real Name': [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(num_samples)],
    'Followers': np.random.randint(50, 50000, num_samples),
    'Following': np.random.randint(10, 5000, num_samples),
    'Posts': np.random.randint(0, 500, num_samples),
    'Engagement Rate (%)': np.random.uniform(0, 10, num_samples).round(2),
    'Bio Text': [random.choice(bio_texts) for _ in range(num_samples)],  # Filling in all bios with random selections
    'Profile Picture': np.random.choice([1, 0], num_samples, p=[0.9, 0.1]),  # 1 for present, 0 for absent
    'Verified': np.random.choice([1, 0], num_samples, p=[0.1, 0.9]),  # 1 for verified, 0 for not verified
    'Account Type': np.random.choice(['Real', 'Fake'], num_samples, p=[0.5, 0.5]),  # Randomly assign real or fake
    
    # Advanced features
    'Account Age (Years)': [(datetime.now() - random_date()).days // 365 for _ in range(num_samples)],  # Calculating age in years
    'Avg Likes per Post': np.random.randint(0, 1000, num_samples),
    'Avg Comments per Post': np.random.randint(0, 100, num_samples),
    'Engagement Consistency (%)': np.random.uniform(50, 100, num_samples).round(2),  # Consistency of engagement
    'Language': [random.choice(['English', 'Spanish', 'French', 'German', 'Japanese']) for _ in range(num_samples)],
    'Profile Link': np.random.choice([1, 0], num_samples, p=[0.6, 0.4])  # 1 for link present, 0 for absent
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('advanced_instagram_fake_real_data_filled_bio.csv', index=False)

print("Advanced dummy dataset with filled 'Bio Text' created as 'dataset.csv'")
