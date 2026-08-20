import warnings; warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv('ad_click_cold_start_with_category.csv')

# Fill missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['gender'].fillna('Unknown', inplace=True)
df['device_type'].fillna('Unknown', inplace=True)
df['ad_position'].fillna('Unknown', inplace=True)
df['time_of_day'].fillna('Unknown', inplace=True)
df['ad_category'].fillna('Unknown', inplace=True)

# Encode
gender_enc   = LabelEncoder(); df['gender_enc']   = gender_enc.fit_transform(df['gender'])
device_enc   = LabelEncoder(); df['device_enc']   = device_enc.fit_transform(df['device_type'])
position_enc = LabelEncoder(); df['position_enc'] = position_enc.fit_transform(df['ad_position'])
time_enc     = LabelEncoder(); df['time_enc']     = time_enc.fit_transform(df['time_of_day'])
category_enc = LabelEncoder(); df['category_enc'] = category_enc.fit_transform(df['ad_category'])
# Feature selection 
X = df[['age','gender_enc','device_enc','position_enc','time_enc','category_enc']]
y = df['click']

# Random Forest gives smooth probabilities (not hard 0/100%)
model = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=10, random_state=42)
model.fit(X, y)

pickle.dump(model,        open('cold_start_model.pkl','wb'))
pickle.dump(gender_enc,   open('gender_encoder.pkl','wb'))
pickle.dump(device_enc,   open('device_encoder.pkl','wb'))
pickle.dump(position_enc, open('position_encoder.pkl','wb'))
pickle.dump(time_enc,     open('time_encoder.pkl','wb'))
pickle.dump(category_enc, open('category_encoder.pkl','wb'))

print("Model trained and saved!")
