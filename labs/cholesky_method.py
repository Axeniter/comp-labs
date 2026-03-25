import numpy as np


def cholesky_method(A: np.ndarray, b: np.ndarray, log=True) -> np.ndarray:
    """Метод квадратного корня (метод Холецкого)"""
    n = len(b)
    A = A.astype(complex)
    S = np.zeros((n,n), dtype=complex)

    print("\nМетод квадратного корня")
    print("="*50)

    for i in range(n):
        k_sum_diag = np.dot(S[:i, i], S[:i, i])
        S[i, i] = np.sqrt(A[i, i] - k_sum_diag)
        
        k_sum_off = np.dot(S[:i, i], S[:i, i+1:])
        S[i, i+1:] = (A[i, i+1:] - k_sum_off) / S[i, i]
                
        if log:
            print(f"Шаг {i}")
            print("-"*50)
            print("Матрица S:")
            print(S)
            print("-"*50)

    y = np.zeros(n, dtype=complex)
    for i in range(n):
        k_sum = np.dot(S[:i, i], y[:i])
        y[i] = (b[i] - k_sum) / S[i, i]
    
    x = np.zeros(n, dtype=complex)
    for i in range(n-1, -1, -1):
        k_sum = np.dot(S[i, i+1:], x[i+1:])
        x[i] = (y[i] - k_sum) / S[i, i]

    return x