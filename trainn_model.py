import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_excel(r"C:\Users\poojitha\Downloads\medical dataset analysis.xlsx")

# Combine Symptoms
df["Combined_Text"] = (
    df["Symptom_1"].fillna("").astype(str) + " " +
    df["Symptom_2"].fillna("").astype(str) + " " +
    df["Symptom_3"].fillna("").astype(str) + " " +
    df["Symptom_4"].fillna("").astype(str) + " " +
    df["Symptom_5"].fillna("").astype(str)
)

# TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1,2),
    max_features=5000
)

X = vectorizer.fit_transform(df["Combined_Text"])

y = df["Disease"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy =", accuracy_score(y_test, pred))

joblib.dump(model, "disease_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model Saved Successfully")