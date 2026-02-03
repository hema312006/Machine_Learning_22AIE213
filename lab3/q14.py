# A14 : Classification using Matrix Inversion Technique

import pandas as pd
import numpy as np

# Load dataset
data = pd.read_excel(
    r"C:\Users\hema3\Downloads\machine learning\Lab Session Data.xlsx",
    sheet_name="Purchase data"
)

# Feature matrix
input_features = data[
    ['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']
].values

# Convert class labels to numeric form
# RICH → 1, POOR → 0
target_numeric = np.array(
    [1 if label > 200 else 0 for label in data['Payment (Rs)']]
)

# Compute weight vector using pseudo-inverse
# This is a least-squares linear model
weight_vector = np.linalg.pinv(input_features) @ target_numeric

print("Matrix Inversion Technique Result:")
print("Weight vector:", weight_vector)
