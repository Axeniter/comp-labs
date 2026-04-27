import numpy as np
from utils import check_residual


def richardson_method(A, b, tau=None, tol=1e-10, log=True):
    """Метод Ричардсона"""
    n = len(b)
    x = np.zeros(n)

    tau = compute_optimal_tau(A) if tau is None else tau
    
    r = A @ x - b
    r_norm = np.linalg.norm(r)
    
    if log:
        k = 0 
        print(f"Приближение k = {k}")
        print("-"*50)
        print(f"Вектор x: {x}")
        print(f"Норма невязки: {r_norm}")
        print("-"*50)

    while r_norm >= tol:
        x = x - tau * r
        
        r = A @ x - b
        r_norm = np.linalg.norm(r)
        
        if log:
            k += 1
            print(f"Приближение k = {k}")
            print("-"*50)
            print(f"Вектор x: {x}")
            print(f"Норма невязки: {r_norm}")
            print("-"*50)
    
    return x


def check_richardson_convergence(A: np.ndarray, tau: float) -> bool:
    """Проверяет сходимость метода Ричардсона по критерию сходимости |λ(S)| < 1"""
    n = A.shape[0]
    S = np.eye(n) - tau * A
    
    rho = np.max(np.abs(np.linalg.eigvals(S)))
    
    return rho < 1.0


def compute_optimal_tau(A):
    """Вычисление оптимального tau для симметричной положительно определённой матрицы"""
    eigenvalues = np.linalg.eigvalsh(A)
    
    lambda_min = np.min(eigenvalues)
    lambda_max = np.max(eigenvalues)
    
    tau_opt = 2 / (lambda_min + lambda_max)
    
    return tau_opt