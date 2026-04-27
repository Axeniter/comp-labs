import numpy as np
from utils import check_residual


def sor_method(A: np.ndarray, b: np.ndarray, omega=1.0, x0=None, tol=1e-10, log=True) -> np.ndarray:
    """Метод последовательной релаксации (SOR)"""
    n = len(b)
    x = np.zeros(n, dtype=float) if x0 is None else np.array(x0, dtype=float)

    if log:
        k = 0
        print(f"Приближение k = {k}")
        print("-"*50)
        print(f"Вектор x: {x}")
        print(f"Норма невязки: {check_residual(A, b, x)}")
        print("-"*50)
    
    while True:
        x_old = x.copy()
        
        for i in range(n):
            sum_1 = np.dot(A[i, :i], x[:i])
            sum_2 = np.dot(A[i, i+1:], x_old[i+1:])
            sum_total = sum_1 + sum_2
            
            x[i] = (1 - omega) * x_old[i] + (omega / A[i, i]) * (b[i] - sum_total)
        
        r_norm = check_residual(A, b, x)
        
        if log:
            k += 1
            print(f"Приближение k = {k}")
            print("-"*50)
            print(f"Вектор x: {x}")
            print(f"Норма невязки: {r_norm:.6e}")
            print("-"*50)
        
        if r_norm < tol:
            break
    
    return x


def check_sor_convergence(A: np.ndarray, omega=1.0) -> bool:
    """Проверяет сходимость метода SOR по критерию сходимости |λ(S)| < 1"""

    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    
    S = np.linalg.inv(D + omega * L) @ ((1 - omega) * D - omega * U)
    rho = np.max(np.abs(np.linalg.eigvals(S)))
    
    return rho < 1.0