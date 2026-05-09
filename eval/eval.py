import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

print("✅ Evaluation Started")

# ==============================
# 1. CHECK PATH
# ==============================
print("📂 Current Folder:", os.getcwd())

# ==============================
# 2. LOAD DATA
# ==============================
try:
    df = pd.read_csv("ad_click_cold_start_with_category.csv")
    print("✅ Data Loaded")
except Exception as e:
    print("❌ Error loading CSV:", e)
    exit()

# ==============================
# 3. LOAD MODEL & ENCODERS
# ==============================
try:
    model = joblib.load("cold_start_model.pkl")
    le_gender = joblib.load("gender_encoder.pkl")
    le_device = joblib.load("device_encoder.pkl")
    le_position = joblib.load("position_encoder.pkl")
    le_time = joblib.load("time_encoder.pkl")
    le_category = joblib.load("category_encoder.pkl")
    print("✅ Model Loaded")
except Exception as e:
    print("❌ Error loading model files:", e)
    exit()
    

# ==============================
# 4. ENCODING
# ==============================
try:
    df['gender_enc'] = le_gender.transform(df['gender'])
    df['device_enc'] = le_device.transform(df['device_type'])
    df['position_enc'] = le_position.transform(df['ad_position'])
    df['time_enc'] = le_time.transform(df['time_of_day'])
    df['category_enc'] = le_category.transform(df['ad_category'])
    print("✅ Encoding Done")
except Exception as e:
    print("❌ Encoding Error:", e)
    exit()

# ==============================
# 5. FEATURES & TARGET
# ==============================
try:
    X = df[['age','gender_enc','device_enc','position_enc','time_enc','category_enc']]
    y = df['click']
    print("✅ Features Prepared")
except Exception as e:
    print("❌ Feature Error:", e)
    exit()

# ==============================
# 6. PREDICTION
# ==============================
try:
    y_pred = model.predict(X)
    print("✅ Prediction Done")
except Exception as e:
    print("❌ Prediction Error:", e)
    exit()

# ==============================
# 7. RESULTS
# ==============================
print("\n📊 RESULTS:\n")

# Accuracy
print("Accuracy:", accuracy_score(y, y_pred))

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y, y_pred))

# ==============================
# 8. CONFUSION MATRIX
# ==============================
cm = confusion_matrix(y, y_pred)
print("\nConfusion Matrix:\n", cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")   # saved image
plt.show()

# ==============================
# 9. FEATURE IMPORTANCE
# ==============================
importances = model.feature_importances_
features = X.columns

plt.figure()
plt.bar(features, importances)
plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.savefig("feature_importance.png")  # saved image
plt.show()

print("\n✅ DONE! Images saved:")
print("➡ confusion_matrix.png")
print("➡ feature_importance.png")