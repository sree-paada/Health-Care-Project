import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier 
from imblearn.over_sampling import SMOTE 

# --- 1. Data Loading ---
FILE_PATH = 'diabetes.csv'
df = pd.read_csv(FILE_PATH)

# --- 2. Data Preprocessing (Median Imputation) ---
cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df_clean = df.copy()

for col in cols_to_impute:
    median_val = df_clean[df_clean[col] != 0][col].median()
    df_clean[col] = df_clean[col].replace(0, median_val)


# --- 3. Targeted Feature Engineering ---
# 3a. Insulin Resistance Proxy: Glucose / Insulin Ratio
# Note: A small constant is added to avoid potential division errors.
df_clean['Insulin_Resistance_Index'] = df_clean['Glucose'] / (df_clean['Insulin'] + 1e-6)

# 3b. High Risk Combination Flag: BMI >= 30 AND Age >= 40 (Common clinical risk group)
df_clean['High_Risk_Combo'] = ((df_clean['BMI'] >= 30) & (df_clean['Age'] >= 40)).astype(int)

# --- 4. Model Setup and Training ---

X = df_clean.drop('Outcome', axis=1)
y = df_clean['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

model = XGBClassifier(
    learning_rate=0.1,
    max_depth=5,
    n_estimators=50,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train_smote, y_train_smote)
y_pred = model.predict(X_test_scaled)

# --- 5. Evaluation ---
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)
cm = confusion_matrix(y_test, y_pred)

print(f"\n{'='*65}\nFINAL MODEL: XGBoost with Engineered Features\n{'='*65}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Recall (Class 1): {recall:.4f}")
print(f"Test F1-Score (Class 1): {f1:.4f}")
print(f"False Negatives (FN): {cm[1, 0]}")