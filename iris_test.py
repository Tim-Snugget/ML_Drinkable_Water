from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd
import numpy as np

y_test = pd.read_csv('./misc/y_test.csv')

feature_list = [10.0, 200.0, 20000.0, 7.0, 250.0, 753.34, 28.30, 124.0, 3.0]
single_pred = np.array(feature_list).reshape(1, -1)

loaded_model = pickle.load(open("./models/Random_Forest.pkl", "rb"))

# predict
y_true = y_test
y_pred = loaded_model.predict(single_pred)
print('iris classifier: accuracy:', accuracy_score(y_true, y_pred))
# iris classifier: accuracy: 0.9333333333333333
