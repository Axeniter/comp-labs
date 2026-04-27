import numpy as np


def richardson_method(A, b, tau=None, tol=1e-10):
    """Метод Ричардсона"""
    n = len(b)
    x = np.zeros(n)

    tau = compute_optimal_tau(A) if tau is None else tau
    
    while True:
        r = A @ x - b

        if np.linalg.norm(r) < tol:
            break
        
        x_new = x - tau * r
        x = x_new
    
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