import numpy as np
import pandas as pd

#loading data
def load_data(file_path):
    data = pd.read_excel(file_path, sheet_name="Purchase data")
    X = data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    y = data["Payment (Rs)"].values.reshape(-1, 1)
    return X, y

#to find rank 
def matrix_rank(matrix):
    return np.linalg.matrix_rank(matrix)

#to calculate cost of each product using pseudo inverse method
def find_cost(matrix, price):
    pseudo_inverse = np.linalg.pinv(matrix)
    cost = np.matmul(pseudo_inverse, price)
    return cost

def main():
    file_path = r"C:\Users\hema3\Downloads\machine learning\Lab Session Data.xlsx"
    X, y = load_data(file_path)
        
    #rank
    rank_X = matrix_rank(X)
    #Cost
    cost = find_cost(X, y)

    #display output
    print("Rank:", rank_X)    
    print("Candies:", cost[0][0])
    print("Mangoes:", cost[1][0])
    print("Milk:", cost[2][0])


if __name__ == "__main__":
    main()
