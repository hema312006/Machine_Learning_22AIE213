# Q8 : Accuracy of kNN Classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_excel(r"C:\Users\hema3\Downloads\machine learning\Lab Session Data.xlsx", sheet_name="Purchase data")

# Features and labels
X = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
y = df['Payment (Rs)'].apply(lambda x: "RICH" if x > 200 else "POOR").values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Calculate accuracy
accuracy = knn.score(X_test, y_test)

print("Accuracy of kNN Classifier:", accuracy)
