import numpy as np
from utils import print_system

def optimal_exclusion(A, b):
    """Метод оптимального исключения"""
    n = len(b)
    
    for k in range(n):
        max_row = k + np.argmax(np.abs(A[k:, k]))
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
            b[[k, max_row]] = b[[max_row, k]]
        
        if k > 0:
            coeffs = A[k, :k].copy()
            for i in range(k):
                A[k, :] -= coeffs[i] * A[i, :]
                b[k] -= coeffs[i] * b[i]
        
        pivot = A[k, k]
        A[k, :] /= pivot
        b[k] /= pivot
        
        for i in range(k):
            d = A[i, k]
            if d != 0:
                A[i, :] -= d * A[k, :]
                b[i] -= d * b[k]
        
        print_system(A, b)
    
    return b