import numpy as np
from utils import check_residual


def jacobi_method(A: np.ndarray, b: np.ndarray, x0=None, tol=1e-10, max_iter=5000, log=True) -> np.ndarray:
    """Метод Якоби"""
    n = len(b)
    D = np.diag(A)
    D_inv = 1.0 / D

    A_LR = A - np.diag(D)

    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.copy(x0).astype(float)

    if log:
            print(f"Приближение k = {0}")
            print("-"*50)
            print(f"Вектор x: {x}")
            print(f"Норма невязки: {check_residual(A, b, x)}")
            print("-"*50)


    for k in range(1, max_iter+1):
        x_new = D_inv * (b - A_LR @ x)
        x = x_new

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


def check_jacobi_convergence(A: np.ndarray) -> bool:
    """Проверяет сходимость метода Якоби по критерию сходимости |λ(S)| < 1"""
    D = np.diag(A)
    
    D_inv = 1.0 / D
    A_LR = A - np.diag(D)
    S = -D_inv[:, np.newaxis] * A_LR
    
    rho = np.max(np.abs(np.linalg.eigvals(S)))
    
    return rho < 1.0