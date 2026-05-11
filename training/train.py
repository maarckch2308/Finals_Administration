from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

X, y = load_iris(return_X_y=True)

model = RandomForestClassifier()
model.fit(X, y)

os.makedirs("/model", exist_ok=True)

with open("/model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved")