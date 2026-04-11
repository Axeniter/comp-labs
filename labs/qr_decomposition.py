import numpy as np


def qr_decomposition(A: np.ndarray, b: np.ndarray, log=True):
    """Метод отражений (QR-разложение)"""
    n = len(b)

    print("\nМетод отражений")
    print("="*50)

    for k in range(n - 1):
        a_norm = np.linalg.norm(A[k:, k])
        p = np.concatenate(([0]*k, A[k:, k]))
        sigma = 1 if A[k,k] >= 0 else -1
        p[k] += sigma * a_norm

        p_norm_squared = np.dot(p[k:], p[k:])
        
        A[k, k] = -sigma * a_norm
        A[k+1:, k] = 0.0
        for j in range(k+1, n):
            A[k:, j] -= (2 * (p[k:] @ A[k:, j]) / p_norm_squared) * p[k:]
        b[k:] -= (2 * (p[k:] @ b[k:]) / p_norm_squared) * p[k:]

        if log:
            print(f"Шаг {k}")
            print("-"*50)
            print("Матрица A:")
            print(A)
            print("Вектор b:")
            print(b)
            print("-"*50)

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i+1:] @ x[i+1:]) / A[i, i]
    return x
        