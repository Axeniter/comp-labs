import numpy as np
from utils import check_residual


def sor_method(A: np.ndarray, b: np.ndarray, omega=1.0, x0=None, tol=1e-10, log=True):
    """Метод последовательной релаксации (SOR)"""
    n = len(b)
    x = np.zeros(n, dtype=float) if x0 is None else np.array(x0, dtype=float)

    if log:
        k = 0
        print(f"Приближение k = {k}")
        print("-"*50)
        print(x)
        print("-"*50)
    
    while check_residual(A, b, x) > tol:
        x_old = x.copy()
        
        for i in range(n):
            sum_1 = np.dot(A[i, :i], x[:i])
            sum_2 = np.dot(A[i, i+1:], x_old[i+1:])
            sum_total = sum_1 + sum_2
            
            x[i] = (1 - omega) * x_old[i] + (omega / A[i, i]) * (b[i] - sum_total)
        
        if log:
            k += 1
            print(f"Приближение k = {k}")
            print("-"*50)
            print(x)
            print("-"*50)
    
    return x


def check_sor_convergence(A, omega=1.0):
    """Проверяет сходимость метода SOR по критерию сходимости |λ(S)| < 1"""

    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    
    S = np.linalg.inv(D + omega * L) @ ((1 - omega) * D - omega * U)
    rho = np.max(np.abs(np.linalg.eigvals(S)))
    
    return rho < 1.0