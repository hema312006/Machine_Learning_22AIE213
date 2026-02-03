# A10 : Manual kNN Classification

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Read Excel file
data_frame = pd.read_excel(
    r"C:\Users\hema3\Downloads\machine learning\Lab Session Data.xlsx",
    sheet_name="Purchase data"
)

# Extract feature matrix
feature_data = data_frame[
    ['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']
].values

# Create class labels based on payment amount
class_labels = data_frame['Payment (Rs)'].apply(
    lambda amount: "RICH" if amount > 200 else "POOR"
).values

X_train, X_test, y_train, y_test = train_test_split(
    feature_data,
    class_labels,
    test_size=0.3
)
def manual_knn_classifier(train_features, train_labels, test_sample, k_value):
    """
    Predict the class of a test sample using manual kNN logic
    """
    distance_label_pairs = []

    # Compute distance between test sample and all training samples
    for index in range(len(train_features)):
        distance = np.linalg.norm(train_features[index] - test_sample)
        distance_label_pairs.append((distance, train_labels[index]))

    # Sort based on distance (ascending)
    distance_label_pairs.sort(key=lambda pair: pair[0])

    # Select k nearest neighbors
    k_nearest = distance_label_pairs[:k_value]

    # Extract labels of nearest neighbors
    neighbor_labels = [label for _, label in k_nearest]

    # Return the most frequent label
    return max(set(neighbor_labels), key=neighbor_labels.count)

print("Manual kNN predictions (k = 3):")

for test_sample in X_test:
    predicted_class = manual_knn_classifier(
        X_train,
        y_train,
        test_sample,
        3
    )
    print(predicted_class)
