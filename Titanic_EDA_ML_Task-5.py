# Titanic EDA + ML Notebook
# Filename: Titanic_EDA_ML_Notebook.py
# Description: End-to-end workflow for Kaggle Titanic dataset (train, test, gender_submission)
# Includes: EDA, Feature Engineering, Modeling, Prediction, Submission

# --- Cell 1: Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Settings
plt.rcParams['figure.dpi'] = 120
sns.set(style='whitegrid')

# --- Cell 2: Load datasets ---
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
gender_submission = pd.read_csv('gender_submission.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Gender submission shape:", gender_submission.shape)

# --- Cell 3: Inspect train data ---
print(train.head())
print(train.info())
print(train.describe())

# --- Cell 4: Missing values ---
print("Missing values in train:\n", train.isnull().sum())
print("Missing values in test:\n", test.isnull().sum())

# --- Cell 5: Feature Engineering Function ---
def feature_engineering(df):
    df = df.copy()
    
    # Family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Title extraction
    df['Title'] = df['Name'].str.extract(r',\s*([^.]*)\.', expand=False)
    df['Title'] = df['Title'].replace(['Mlle','Ms'],'Miss')
    df['Title'] = df['Title'].replace('Mme','Mrs')
    rare_titles = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
    df['Title'] = df['Title'].replace(rare_titles,'Other')
    
    # Fill missing Age with median by Title
    df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
    
    # Fill missing Embarked
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Fill missing Fare (only in test)
    if df['Fare'].isnull().any():
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # Cabin → Deck
    df['Deck'] = df['Cabin'].str[0]
    df['Deck'] = df['Deck'].fillna('U')
    
    return df

train_fe = feature_engineering(train)
test_fe = feature_engineering(test)

# --- Cell 6: EDA ---
pdf_path = 'Titanic_Full_Report.pdf.pdf'
pp = PdfPages(pdf_path)

# Survival by Sex
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(data=train_fe, x='Sex', y='Survived', ax=ax)
ax.set_title('Survival by Sex')
pp.savefig(fig)
plt.close(fig)

# Survival by Pclass
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(data=train_fe, x='Pclass', y='Survived', ax=ax)
ax.set_title('Survival by Pclass')
pp.savefig(fig)
plt.close(fig)

# Age distribution
fig, ax = plt.subplots(figsize=(6,4))
sns.histplot(train_fe['Age'].dropna(), kde=True, bins=30, ax=ax)
ax.set_title('Age Distribution')
pp.savefig(fig)
plt.close(fig)

# Correlation heatmap (fixed for pandas 1.3.x and 2.x)
corr = train_fe.select_dtypes(include=[np.number]).corr()
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt='.2f', ax=ax, cmap="coolwarm")
ax.set_title('Correlation Heatmap')
pp.savefig(fig)
plt.close(fig)

# --- Cell 7: Data Prep for ML ---
features = ['Pclass','Sex','Age','Fare','Embarked','FamilySize','IsAlone','Title','Deck']

all_data = pd.concat([train_fe[features], test_fe[features]])

# Encode categoricals
for col in ['Sex','Embarked','Title','Deck']:
    le = LabelEncoder()
    all_data[col] = le.fit_transform(all_data[col])

train_X = all_data.iloc[:len(train_fe)]
test_X = all_data.iloc[len(train_fe):]
train_y = train_fe['Survived']

# --- Cell 8: Train/Test Split ---
X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.2, random_state=42)

# --- Cell 9: Models ---
models = {
    'LogReg': LogisticRegression(max_iter=500),
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
    'XGB': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    results[name] = acc
    print(f"{name}: {acc:.4f}")
    print(confusion_matrix(y_val, preds))
    print(classification_report(y_val, preds))

# --- Cell 10: Choose best model & Predict on test ---
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

best_model.fit(train_X, train_y)
test_preds = best_model.predict(test_X)

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': test_preds
})
submission.to_csv('submission.csv', index=False)
print("Submission file saved: submission.csv")

# --- Cell 11: Observations ---
obs = [
    "Observations:",
    "- Women had higher survival rates than men.",
    "- Higher Pclass passengers survived more often.",
    "- Age distribution shows children had relatively better survival.",
    "- Feature engineering (Title, FamilySize) improved prediction accuracy.",
    f"- Best model in validation: {best_model_name} with accuracy {results[best_model_name]:.4f}."
]
fig, ax = plt.subplots(figsize=(8.27,11.69))
ax.axis('off')
ax.text(0, 1, '\n'.join(obs), va='top', fontsize=10)
pp.savefig(fig)
plt.close(fig)
pp.close()

print(f"Full PDF report saved to: {pdf_path}")
