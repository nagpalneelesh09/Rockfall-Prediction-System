import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report 

df = pd.read_csv("dataset/rockfall_data.csv")

X = df.drop(["Risk_Level", "Risk_Score"], axis=1)

y = df["Risk_Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy * 100, "%")

print(classification_report(y_test, predictions))

joblib.dump(model, "model/rockfall_model.pkl")

print("Model saved successfully!")

