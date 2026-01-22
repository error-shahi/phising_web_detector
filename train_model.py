import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from features import extract_features

print("📥 Loading dataset...")
df = pd.read_csv("dataset.csv")

# Ensure correct column names
df.columns = [c.lower() for c in df.columns]

# Keep only required columns
df = df[['url', 'label']]

# Drop missing values
df.dropna(inplace=True)

print(f"✅ Total samples: {len(df)}")

# Feature extraction
print("⚙️ Extracting features...")
X = df['url'].apply(extract_features).tolist()
y = df['label'].tolist()

# Train / Test split (IMPORTANT)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("🧠 Training model...")

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluation
print("\n📊 MODEL EVALUATION\n")

y_pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred), "\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "phishing_model.pkl")

print("\n✅ High-accuracy model saved as phishing_model.pkl")
