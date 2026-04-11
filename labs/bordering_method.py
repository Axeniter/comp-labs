import numpy as np


def bordering_method(A: np.ndarray, b: np.ndarray, log=True) -> np.ndarray:
    """Решение системы с использованием метода окаймления для нахождения обратной матрицы"""
    n = len(b)
    
    print("\nМетод окаймления")
    print("="*50)

    A_inv = np.zeros((n,n))
    A_inv[0, 0] = 1 / A[0, 0]

    if log:
            print(f"Шаг {0}")
            print("-"*50)
            print("Обратная матрица:")
            print(A_inv)
            print("-"*50)

    for i in range(1,n):
        v = A[i, :i]
        u = A[:i, i]
        alpha = A[i, i] - v @ A_inv[:i,:i] @ u

        r = - (A_inv[:i, :i] @ u) / alpha
        q = - (v @ A_inv[:i, :i]) / alpha

        A_inv[:i, i] = r
        A_inv[i, :i] = q
        A_inv[i, i] = 1 / alpha
        A_inv[:i, :i] += alpha * np.outer(r, q)

        if log:
            print(f"Шаг {i}")
            print("-"*50)
            print("Обратная матрица:")
            print(A_inv)
            print("-"*50)

    x = A_inv @ b
    return x