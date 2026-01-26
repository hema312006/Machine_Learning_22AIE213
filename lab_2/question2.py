import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

#load data
def load_data(file_path):
    data = pd.read_excel(file_path, sheet_name="Purchase data")
    X = data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    payment = data["Payment (Rs)"].values
    return X, payment

#generate labels based on payment
def labels_generation(payment):
    labels = []
    for p in payment:
        if p > 200:
            labels.append(1)   
        else:
            labels.append(0)
    return np.array(labels)

def main():
    #file path
    file_path = r"C:\Users\hema3\Downloads\machine learning\Lab Session Data.xlsx"

    X, payment = load_data(file_path)
    y = labels_generation(payment)
    #train classifier model
    model = LogisticRegression()
    model.fit(X, y)
    #predictions
    predictions = model.predict(X)

    #diaplay output
    for i in range(len(predictions)):
        label = "RICH" if predictions[i] == 1 else "POOR"
        print("Customer", i + 1, "is classified as", label)


if __name__ == "__main__":
    main()
