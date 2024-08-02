import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import datetime 
import time
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model, metrics
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import sys
import os
os.system('chcp 65001')
sys.stdout.reconfigure(encoding='utf-8')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

path = r"C:\Users\HP\OneDrive\Desktop\python\train.csv"
data_train = pd.read_csv(path)

print(data_train.head())
print(data_train.shape) 
# (14999, 9)

category = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8}
data_train["category"] = data_train["category"].map(category)
#print(data_train.head())

data_train = data_train.query("views != 'F' and likes != 'F' and dislikes != 'F' and comment != 'F'")

data_train["adview"] = pd.to_numeric(data_train["adview"])
data_train["views"] = pd.to_numeric(data_train["views"])
data_train["likes"] = pd.to_numeric(data_train["likes"])
data_train["dislikes"] = pd.to_numeric(data_train["dislikes"])
data_train["comment"] = pd.to_numeric(data_train["comment"])

label_encoder = LabelEncoder()
data_train['vidid'] = label_encoder.fit_transform(data_train['vidid'])
data_train['duration'] = label_encoder.fit_transform(data_train['duration'])
data_train['published'] = label_encoder.fit_transform(data_train['published'])

print("Label Encoded Data:")
#print(data_train.head())

def checki(x):
    y = x[2:]  
    h = ''
    m = ''
    s = ''
    mm = ''
    P = ['H', 'M', 'S']
    for i in y:
        if i not in P:
            mm += i
        else:
            if i == "H":
                h = mm
                mm = ''
            elif i == "M":
                m = mm
                mm = ''
            else:
                s = mm
                mm = ''
    if h == '':
        h = '00'
    if m == '':
        m = '00'
    if s == '':
        s = '00'
    bp = h + ':' + m + ':' + s
    return bp

train = pd.read_csv(path)
mp = pd.read_csv(path)["duration"]

time = mp.apply(checki)
def func_sec(time_string):
    h, m, s = time_string.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)
time1=time.apply(func_sec)
data_train["duration"]=time1
data_train.head()
print(data_train)


plt.hist(data_train["category"])
plt.title('Category Distribution')
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.show()


plt.plot(data_train["adview"])
plt.title('Adview Counts')
plt.xlabel('Index')
plt.ylabel('Adviews')
plt.show()

data_train = data_train[data_train["adview"] < 2000000]
numeric_data_train = data_train.select_dtypes(include=[np.number])
f, ax = plt.subplots(figsize=(10, 8))
corr = numeric_data_train.corr()
sns.heatmap(corr, annot=True)
plt.show()
# Split data
Y_train = pd.DataFrame(data=data_train.iloc[:, 1].values, columns=['target'])
data_train = data_train.drop(["adview", "vidid"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(data_train, Y_train, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Mean of X_train after standardization:", X_train.mean())
print("Standard deviation of X_train after standardization:", X_train.std())
#y_train and y_test are 1D arrays
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

def print_error(X_test, y_test, model):
    prediction = model.predict(X_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, prediction))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, prediction))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
    print('R-squared:', metrics.r2_score(y_test, prediction))

# Linear Regression
linear_regression = linear_model.LinearRegression()
linear_regression.fit(X_train, y_train)
print("\nLinear Regression Results:")
print_error(X_test, y_test, linear_regression)

# Support Vector Regressor
supportvector_regressor = SVR()
supportvector_regressor.fit(X_train, y_train)
print("\nSupport Vector Regressor Results:")
print_error(X_test, y_test, supportvector_regressor)

# Decision Tree Regressor
decision_tree = DecisionTreeRegressor()
decision_tree.fit(X_train, y_train)
print("\nDecision Tree Regressor Results:")
print_error(X_test, y_test, decision_tree)

# Random Forest Regressor
random_forest = RandomForestRegressor(n_estimators=200, max_depth=25, min_samples_split=15, min_samples_leaf=2, random_state=42)
random_forest.fit(X_train, y_train)
print("\nRandom Forest Regressor Results:")
print_error(X_test, y_test, random_forest)

# ANN Model
ann = Sequential([
    Dense(6, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(6, activation="relu"),
    Dense(1)
])

optimizer = Adam()
ann.compile(optimizer=optimizer, loss=MeanSquaredError(), metrics=["mean_squared_error"])

history = ann.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=1)
ann.summary()

# Evaluate the ANN model
print_error(X_test, y_test, ann)

# Plot learning curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Learning Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

import keras

# Save the ANN model
keras.saving.save_model(ann, 'my_model.keras')

