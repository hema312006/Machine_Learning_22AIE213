import numpy as np

def matrix_multiplication(A, B, c1, r2):
    if c1 != r2:
        return None
    return np.dot(A, B)


r1 = int(input("Enter rows of matrix A: "))
c1 = int(input("Enter columns of matrix A: "))

print("Enter elements of matrix A:")
A = []
for _ in range(r1):
    A.append(list(map(int, input().split())))

r2 = int(input("Enter rows of matrix B: "))
c2 = int(input("Enter columns of matrix B: "))

print("Enter elements of matrix B:")
B = []
for _ in range(r2):
    B.append(list(map(int, input().split())))

result = matrix_multiplication(A, B, c1, r2)

if result is None:
    print("Error: Matrices cannot be multiplied")
else:
    print("Product of matrices A and B:")
    for row in result:
        print(row)
