import numpy as np
from utils import print_system


def optimal_exclusion(A: np.ndarray, b: np.ndarray, log=True) -> np.ndarray:
    """Метод оптимального исключения"""
    n = len(b)
    
    print("\nМетод оптимального исключения")
    print("="*50)

    for k in range(n):
        max_row = k + np.argmax(np.abs(A[k:, k]))
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
            b[[k, max_row]] = b[[max_row, k]]
        
        if k > 0:
            for i in range(k):
                d = A[k, i]
                if d != 0:
                    A[k, :] -= d * A[i, :]
                    b[k] -= d * b[i]
        
        pivot = A[k, k]
        A[k, :] /= pivot
        b[k] /= pivot
        
        for i in range(k):
            d = A[i, k]
            if d != 0:
                A[i, :] -= d * A[k, :]
                b[i] -= d * b[k]
        
        if log:
            print(f"Шаг {k}")
            print("-"*50)
            print_system(A, b)
            print("-"*50)
    
    return b