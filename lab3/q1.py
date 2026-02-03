import numpy as np

#Dot vector
def dot_product(A, B):
    result=0
    for i in range(len(A)):
        result += A[i] * B[i]
    return result

##euclidean norm of vector A
def euclidean_norm(A):
    sum = 0
    for i in range(len(A)):
        sum += A[i]**2
    return sum ** 0.5

n=int(input("enter the num of dimensions: "))

A=list(map(float, input("enter vector A elements: ").split()))
B=list(map(float, input("enter vector B elements: ").split()))

print("\nManual Dot Product:", dot_product(A, B))
print("NumPy Dot Product:", np.dot(A, B))

print("\nManual Norm of A:", euclidean_norm(A))
print("NumPy Norm of A:", np.linalg.norm(A))

print("\nManual Norm of B:", euclidean_norm(B))
print("NumPy Norm of B:", np.linalg.norm(B))
    
