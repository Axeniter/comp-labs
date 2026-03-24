import numpy as np


def cholesky_method(A: np.ndarray, b: np.ndarray, log=True) -> np.ndarray:
    """Метод квадратного корня (метод Холецкого)"""
    n = len(b)
    A = A.astype(complex)
    S = np.zeros((n,n), dtype=complex)

    print("\nМетод квадратного корня")
    print("="*50)

    for i in range(n):
        for j in range(i, n):
            k_sum = sum(S[k, i] * S[k, j] for k in range(i))

            if i == j:
                S[i, i] = np.sqrt(A[i,i] - k_sum)
            else:
                S[i, j] = (A[i,j] - k_sum) / S[i, i]
                
        if log:
            print(f"Шаг {i}")
            print("-"*50)
            print("Матрица S:")
            print(S)
            print("-"*50)

    y = np.zeros(n, dtype=complex)
    for i in range(n):
        k_sum = sum((S[k,i] * y[k] for k in range(i)))
        y[i] = (b[i] - k_sum) / S[i,i]
    
    x = np.zeros(n, dtype=complex)
    for i in range(n-1, -1, -1):
        k_sum = sum((S[i,k] * x[k] for k in range(i+1, n)))
        x[i] = (y[i] - k_sum) / S[i,i]

    return x