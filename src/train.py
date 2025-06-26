import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import wandb

# W&B config
wandb.init(project="diabetes-decision-tree", config={
    "model": "DecisionTree",
    "test_size": 0.2,
    "random_state": 42
})
config = wandb.config

# Load data
df = pd.read_csv('data/diabetes.csv')
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.test_size, random_state=config.random_state)

# Train
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

wandb.log({"accuracy": acc})

# Save model
joblib.dump(model, 'models/model.joblib')

