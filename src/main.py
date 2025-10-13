import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 2️⃣ Data Loading & Overview
df = pd.read_csv("../data/covtype.csv")
print(df.head())

print('Shape:', df.shape)
print(df.info())
print(df.describe())

print(df['Cover_Type'].value_counts())

cover_type_count = df['Cover_Type'].value_counts().sort_index()
plt.figure(figsize=(8,6))
bars = plt.bar(x=cover_type_count.index, height=cover_type_count.values)
plt.bar_label(bars, fmt='%.0f')
plt.title("Value Counts of Cover Type")
plt.xlabel("Cover Type")
plt.ylabel("Count")
plt.show()

print("Missing values:", df.isna().sum().sum())
print("Duplicates:", df.duplicated().sum())

# 3️⃣ Exploratory Data Analysis (EDA)
num_cols = df.columns[:10].tolist() + ['Cover_Type']
plt.figure(figsize=(12,8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="Blues", fmt=".2f")
plt.title("Correlation Heatmap (Numeric Features + Cover_Type)")
plt.show()

desc_stats = df.describe().T
desc_stats["median"] = df.median()
desc_stats["skewness"] = df.skew()
desc_stats = desc_stats[["mean", "median", "std", "min", "25%", "50%", "75%", "max", "skewness"]]
print(desc_stats.head(10))

# Histograms
df.iloc[:, :10].hist(figsize=(15,10), bins=20, color='skyblue')
plt.suptitle("Histograms of Numeric Features", fontsize=16)
plt.show()

# Boxplots (before outlier handling)
plt.figure(figsize=(15,8))
df.iloc[:, :10].boxplot()
plt.title("Boxplots of Numeric Features (Before Outlier Capping)")
plt.xticks(rotation=90)
plt.show()

# Capping the outliers by replacing extreme values with boundary values
num_features = df.columns[:10]
Q1 = df[num_features].quantile(0.25)
Q3 = df[num_features].quantile(0.75)
IQR = Q3 - Q1
outliers = ((df[num_features] < (Q1 - 1.5 * IQR)) | (df[num_features] > (Q3 + 1.5 * IQR))).sum()
print("Outliers per feature:\n", outliers.sort_values(ascending=False))

# Cap outliers
for col in num_features:
    lower_bound = Q1[col] - 1.5 * IQR[col]
    upper_bound = Q3[col] + 1.5 * IQR[col]
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

# Boxplots after
plt.figure(figsize=(15,8))
df[num_features].boxplot(color='blue')
plt.title("Boxplots of Numeric Features (After Outlier Capping)")
plt.xticks(rotation=90)
plt.show()

wilderness_counts = df[[col for col in df.columns if "Wilderness_Area" in col]].sum()
plt.figure(figsize=(8,5))
sns.barplot(x=wilderness_counts.index, y=wilderness_counts.values, palette="Blues")
plt.title("Distribution of Wilderness Areas")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

soil_counts = df[[col for col in df.columns if "Soil_Type" in col]].sum().sort_values(ascending=False)
plt.figure(figsize=(10,10))
sns.barplot(x=soil_counts.values, y=soil_counts.index)
plt.title("Distribution of Soil Types")
plt.xlabel("Count")
plt.ylabel("Soil Type")
plt.show()

# 4️⃣ Preprocessing & Feature Engineering
# Consolidate Soil Types (from 40 dummies to 1 integer column):
soil_cols = [col for col in df.columns if col.startswith("Soil_Type")]
df["Soil_Type"] = df[soil_cols].idxmax(axis=1).str.replace("Soil_Type", "").astype(int)
df = df.drop(columns=soil_cols)

num_cols = list(num_features) + ['Soil_Type']

print(df.head())

plt.figure(figsize=(15,8))
sns.boxplot(data=df[num_cols], palette='Blues')  # Convert to list here
plt.title("Boxplots After Preprocessing")
plt.xticks(rotation=90)
plt.show()

# 5️⃣ Data Splitting & Scaling
X = df.drop('Cover_Type', axis=1)
y = df['Cover_Type'] - 1  # Shift to 0-6 for XGB

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# 6️⃣ Model Training & Evaluation

# Decision Tree (Baseline):
dt = DecisionTreeClassifier(random_state=42, max_depth=20)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

cm = confusion_matrix(y_test, y_pred_dt)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt.classes_)
disp.plot(cmap="Greens")
plt.title("Decision Tree Confusion Matrix")
plt.show()

# Random Forest (with imbalance handling):
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Random Forest Confusion Matrix")
plt.show()

# XGBoost:
xgb = XGBClassifier(n_estimators=100, random_state=42, max_depth=20, objective='multi:softmax', num_class=7)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

cm = confusion_matrix(y_test, y_pred_xgb)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=xgb.classes_)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("XGBoost Confusion Matrix")
plt.show()

# 7️⃣ Feature Importance

# Random Forest:
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("Top Features (RF):")
for i in range(len(X.columns)):
    print(f"{X.columns[indices[i]]}: {importances[indices[i]]:.4f}")

plt.figure(figsize=(10,6))
plt.bar(range(len(X.columns)), importances[indices])
plt.xticks(range(len(X.columns)), X.columns[indices], rotation=90)
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()

# XGBoost:
importances = xgb.feature_importances_
indices = np.argsort(importances)[::-1]

print("Top Features (XGB):")
for i in range(len(X.columns)):
    print(f"{X.columns[indices[i]]}: {importances[indices[i]]:.4f}")

plt.figure(figsize=(10,6))
plt.bar(range(len(X.columns)), importances[indices], color='skyblue')
plt.xticks(range(len(X.columns)), X.columns[indices], rotation=90)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()

# 8️⃣ Model Comparison
results = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest', 'XGBoost'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_dt),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_xgb)
    ]
})
print(results)

# 9️⃣ (Bonus) Hyperparameter Tuning
params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
grid = RandomizedSearchCV(rf, params, n_iter=5, scoring='accuracy', cv=3, random_state=42)
grid.fit(X_train, y_train)
print("Best Params:", grid.best_params_)
print("Best Accuracy:", grid.best_score_)