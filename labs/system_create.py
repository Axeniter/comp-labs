import numpy as np


def generate_random_system(n, d):
    A = np.random.uniform(-d, d, (n, n))
    b = np.random.uniform(-d, d, n)
    
    return A, b


def generate_symmetrical_system(n, d):
    A, b = generate_random_system(n, d)
    A = np.triu(A) + np.triu(A, 1).T

    return A, b


def create_input_system(n):
    A = np.zeros((n,n))
    b = np.zeros(n)

    for i in range(n):
        print(f"\nУравнение {i + 1}:")
        for j in range(n):
            while True:
                try:
                    value = float(input(f"  a[{i+1}][{j+1}] = "))
                    A[i, j] = value
                    break
                except ValueError:
                    print("(!) Введите число")
        
        while True:
            try:
                value = float(input(f"  b[{i+1}] = "))
                b[i] = value
                break
            except ValueError:
                print("(!) Введите число")
    
    return A, b