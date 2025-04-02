import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score

from sklearn.datasets import fetch_openml

df = fetch_openml(name="creditcard", version=1, as_frame=True)
df = df.frame


# Step 2: Data Preprocessing
# Checking for missing values
df.dropna(inplace=True)


# Step 3: Define Features and Target
X = df.drop(columns=['Class'])  # Independent variables
y = df['Class']  # Target variable (0 = Non-Fraud, 1 = Fraud)

# Convert target labels to integer format
y = y.astype(int)


# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 5: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train an Anomaly Detection Model
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
model.fit(X_train_scaled)

# Step 7: Make Predictions
y_pred = model.predict(X_test_scaled)
y_pred = [1 if p == -1 else 0 for p in y_pred]  # Convert anomaly detection output to fraud/non-fraud

# Step 8: Evaluate the Model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 9: Visualize Fraud vs Non-Fraud Transactions
plt.figure(figsize=(8, 5))
sns.countplot(x=y, palette='coolwarm')
plt.title("Distribution of Fraud vs Non-Fraud Transactions")
plt.show()
