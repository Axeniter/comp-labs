import numpy as np
from utils import check_residual


def gradient_descent(A: np.ndarray, b: np.ndarray, x0=None, tol=1e-10, max_iter=5000, log=True) -> np.ndarray:
    """Метод градиентного спуска"""
    n = len(b)
    x = np.zeros(n, dtype=float) if x0 is None else np.array(x0, dtype=float)

    if log:
        print(f"Приближение k = {0}")
        print("-"*50)
        print(f"Вектор x: {x}")
        print(f"Норма невязки: {check_residual(A, b, x)}")
        print("-"*50)
    
        for k in range(1, max_iter+1):
            r = A @ x - b
            Ar = A @ r
            alpha = np.dot(r, r) / np.dot(r, Ar)
            
            x = x - alpha * r
            
            r_norm = check_residual(A, b, x)
            
            if log:
                print(f"Приближение k = {k}")
                print("-"*50)
                print(f"Вектор x: {x}")
                print(f"Норма невязки: {r_norm}")
                print("-"*50)
            
            if r_norm < tol:
                break
    
    return x


def check_gradient_descent_convergence(A: np.ndarray) -> bool:
    """Проверяет сходимость метода градиентного спуска по критерию сходимости |λ(S)| < 1"""
    eigenvalues = np.linalg.eigvalsh(A)
    
    lambda_min = eigenvalues.min()
    lambda_max = eigenvalues.max()
    
    alpha = 2 / (lambda_min + lambda_max)
    
    S = np.eye(A.shape[0]) - alpha * A
    rho = np.max(np.abs(np.linalg.eigvals(S)))
    
    return rho < 1.0