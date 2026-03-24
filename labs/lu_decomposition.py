import numpy as np


def lu_decomposition(A: np.ndarray, b: np.ndarray):
    """LU-разложение"""
    n = len(b)
    L = np.zeros((n,n))
    U = np.zeros((n,n))

    for i in range(n):
        for j in range(i, n):
            k_sum = sum(L[i,k] * U[k,j] for k in range(i))
            U[i, j] = A[i,j] - k_sum

        for j in range(i, n):
            k_sum = sum(L[j,k] * U[k,i] for k in range(i))
            L[j,i] = (A[j,i] - k_sum) / U[i,i]

    y = np.zeros(n)
    for i in range(n):
        k_sum = sum((L[i,k] * y[k] for k in range(i)))
        y[i] = b[i] - k_sum

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        k_sum = sum((U[i,k] * x[k] for k in range(i+1, n)))
        x[i] = (y[i] - k_sum) / U[i,i]

    return x