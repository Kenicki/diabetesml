import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import wandb

# Initialize wandb project
wandb.init(project="diabetes-classifier", config={"model_type": "DecisionTree"})

# Load data
df = pd.read_csv("../data/diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predictions and accuracy
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Log to wandb
wandb.log({"accuracy": acc})

# Save model
joblib.dump(clf, "../src/model.joblib")
