# A11 : Accuracy of kNN for Different Values of k

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Read the Excel file
data_frame = pd.read_excel(
    r"C:\Users\hema3\Downloads\machine learning\Lab Session Data.xlsx",
    sheet_name="Purchase data"
)

# Extract input features
feature_matrix = data_frame[
    ['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']
].values

# Generate class labels based on payment amount
class_labels = data_frame['Payment (Rs)'].apply(
    lambda amount: "RICH" if amount > 200 else "POOR"
).values

X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix,
    class_labels,
    test_size=0.3
)
accuracy_scores = []

print("A11 Result")
print("----------")
print("Accuracy for different values of k:")

# Maximum k cannot exceed number of training samples
maximum_k = len(X_train)

for k_value in range(1, maximum_k + 1):
    # Initialize kNN classifier for current k
    knn_classifier = KNeighborsClassifier(n_neighbors=k_value)

    # Train the model
    knn_classifier.fit(X_train, y_train)

    # Evaluate accuracy on test data
    current_accuracy = knn_classifier.score(X_test, y_test)
    accuracy_scores.append(current_accuracy)

    print("k =", k_value, "Accuracy =", current_accuracy)

plt.plot(range(1, maximum_k + 1), accuracy_scores, marker='o')
plt.xlabel("k value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k (kNN Classifier)")
plt.show()
