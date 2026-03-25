import numpy as np


def lu_decomposition(A: np.ndarray, b: np.ndarray, log=True) -> np.ndarray:
    """LU-разложение"""
    n = len(b)
    L = np.zeros((n,n))
    U = np.zeros((n,n))

    print("\nРешение LU-разложением")
    print("="*50)

    for i in range(n):
        U[i, i:] = A[i, i:] - np.dot(L[i, :i], U[:i, i:])
        L[i:, i] = (A[i:, i] - np.dot(L[i:, :i], U[:i, i])) / U[i, i]

        if log:
            print(f"Шаг {i}")
            print("-"*50)
            print("Матрица L:")
            print(L)
            print("\nМатрица U:")
            print(U)
            print("-"*50)

    y = np.zeros(n)
    for i in range(n):
        k_sum = np.dot(L[i, :i], y[:i])
        y[i] = b[i] - k_sum

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        k_sum = np.dot(U[i, i+1:], x[i+1:])
        x[i] = (y[i] - k_sum) / U[i, i]
    
    return x