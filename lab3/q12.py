# A12 : Confusion Matrix for kNN Classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Load dataset
data_frame = pd.read_excel(
    r"C:\Users\hema3\Downloads\machine learning\Lab Session Data.xlsx",
    sheet_name="Purchase data"
)

# Extract features
feature_matrix = data_frame[
    ['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']
].values

# Create class labels
class_labels = data_frame['Payment (Rs)'].apply(
    lambda amount: "RICH" if amount > 200 else "POOR"
).values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix,
    class_labels,
    test_size=0.3
)

# Train kNN classifier (k = 3)
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Predict test labels
predicted_labels = knn_model.predict(X_test)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, predicted_labels)

print("Confusion Matrix:")
print(conf_matrix)
