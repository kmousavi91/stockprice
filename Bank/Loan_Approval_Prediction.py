import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Generate synthetic loan approval data
np.random.seed(42)
n_samples = 1000
df = pd.DataFrame({
    'Applicant Income': np.random.randint(20000, 120000, size=n_samples),
    'Loan Amount': np.random.randint(5000, 50000, size=n_samples),
    'Credit Score': np.random.randint(300, 850, size=n_samples),
    'Employment Status': np.random.choice(['Employed', 'Unemployed', 'Self-Employed'], size=n_samples),
    'Debt-to-Income Ratio': np.random.uniform(0.1, 0.5, size=n_samples),
    'Loan Term': np.random.choice([12, 24, 36, 48, 60], size=n_samples),
    'Previous Default': np.random.choice([0, 1], size=n_samples),
    'Loan Approved': np.random.choice([0, 1], size=n_samples)  # Target Variable
})

# Encode categorical variables
label_encoder = LabelEncoder()
df['Employment Status'] = label_encoder.fit_transform(df['Employment Status'])

# Split data into features and target variable
X = df.drop(columns=['Loan Approved'])
y = df['Loan Approved']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
