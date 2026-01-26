import numpy as np
import pandas as pd


# read thyroid data
def read_data(path):
    return pd.read_excel(path, sheet_name="thyroid0387_UCI")


# compute cosine similarity
def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def main():
    path = r"C:\Users\hema3\Downloads\machine learning\Lab Session Data.xlsx"
    df = read_data(path)

    # handle missing values and convert binary flags
    df.replace({"?": 0, "t": 1, "f": 0}, inplace=True)

    # remove non-feature column and ensure numeric data
    features = df.drop(columns=["Record ID"]).apply(pd.to_numeric, errors="coerce").fillna(0)

    # select two samples
    x1, x2 = features.iloc[0].values, features.iloc[1].values

    # similarity score
    score = cosine_similarity(x1, x2)

    print("Cosine similarity (first two samples):", score)


if __name__ == "__main__":
    main()
