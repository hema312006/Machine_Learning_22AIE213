import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# read thyroid dataset
def read_data(path):
    return pd.read_excel(path, sheet_name="thyroid0387_UCI")

# count binary match combinations
def binary_counts(a, b):
    one_one = one_zero = zero_one = zero_zero = 0
    for k in range(len(a)):
        if a[k] == 1 and b[k] == 1:
            one_one += 1
        elif a[k] == 1 and b[k] == 0:
            one_zero += 1
        elif a[k] == 0 and b[k] == 1:
            zero_one += 1
        else:
            zero_zero += 1
    return one_one, one_zero, zero_one, zero_zero

# jaccard similarity
def jaccard_sim(x11, x10, x01):
    denom = x11 + x10 + x01
    return 0 if denom == 0 else x11 / denom

# simple matching similarity
def simple_match(x11, x10, x01, x00):
    total = x11 + x10 + x01 + x00
    return 0 if total == 0 else (x11 + x00) / total

# cosine similarity
def cosine_sim(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def main():
    path = r"C:\Users\hema3\Downloads\machine learning\Lab Session Data.xlsx"
    df = read_data(path)

    # clean missing values and encode binary fields
    df.replace({"?": 0, "t": 1, "f": 0}, inplace=True)

    sample = df.iloc[:20]

    # choose only binary-valued columns
    bin_features = sample.loc[:, sample.nunique() <= 2]
    n = len(bin_features)

    jaccard_mat = np.zeros((n, n))
    smc_mat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            a = bin_features.iloc[i].values
            b = bin_features.iloc[j].values
            x11, x10, x01, x00 = binary_counts(a, b)
            jaccard_mat[i, j] = jaccard_sim(x11, x10, x01)
            smc_mat[i, j] = simple_match(x11, x10, x01, x00)

    # numeric data for cosine similarity
    num_features = sample.drop(columns=["Record ID"]) \
                         .apply(pd.to_numeric, errors="coerce") \
                         .fillna(0)

    cosine_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cosine_mat[i, j] = cosine_sim(
                num_features.iloc[i].values,
                num_features.iloc[j].values
            )

    # heatmaps
    sns.heatmap(jaccard_mat, annot=True, cmap="coolwarm")
    plt.title("Jaccard Similarity")
    plt.show()

    sns.heatmap(smc_mat, annot=True, cmap="coolwarm")
    plt.title("Simple Matching Similarity")
    plt.show()

    sns.heatmap(cosine_mat, annot=True, cmap="coolwarm")
    plt.title("Cosine Similarity")
    plt.show()


if __name__ == "__main__":
    main()
