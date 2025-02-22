import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 🔹 Step 1: Load the dataset
df = pd.read_csv("creditcard.csv")

# 🔹 Step 2: Explore the dataset
print("Dataset Overview:")
print(df.head())

print("\nClass Distribution:")
print(df["Class"].value_counts())

# 🔹 Step 3: Preprocess Data
X = df.drop(columns=["Class"])  # Features
y = df["Class"]  # Target variable (0 = Normal, 1 = Fraud)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 Step 4: Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 🔹 Step 5: Train ML Models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n🔹 {name} Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred))

# 🔹 Step 6: Visualize Fraud vs. Non-Fraud
plt.figure(figsize=(6, 4))
sns.countplot(x="Class", data=df, palette="coolwarm")
plt.title("Fraud vs. Non-Fraud Transactions")
plt.show()
