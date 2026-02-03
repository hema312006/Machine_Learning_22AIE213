# A13 : Performance Metrics from Confusion Matrix

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Load dataset
dataset = pd.read_excel(
    r"C:\Users\hema3\Downloads\machine learning\Lab Session Data.xlsx",
    sheet_name="Purchase data"
)

# Features and labels
X_data = dataset[
    ['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']
].values

y_data = dataset['Payment (Rs)'].apply(
    lambda val: "RICH" if val > 200 else "POOR"
).values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.3
)

# Train kNN
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)

# Predictions
y_predicted = classifier.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_predicted)

# Label order assumed: [POOR, RICH]
tn = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tp = cm[1][1]

# Metric calculations
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * precision * recall / (precision + recall)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
