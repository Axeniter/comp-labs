import numpy as np
from rotation_with_barriers import rotation_with_barriers


def richardson_method(A, b, n_iter=8, tol=1e-10, log=True):
    """Метод Ричардсона с чебышёвскими параметрами"""
    n = len(b)
    x = np.zeros(n)

    tau_0, rho_0, v = compute_parameters(A, n_iter)

    r = A @ x - b
    r_norm = np.linalg.norm(r)

    if log:
        total_iter = 0
        print(f"Приближение k = {total_iter}")
        print("-" * 50)
        print(f"Вектор x: {x}")
        print(f"Норма невязки: {r_norm}")
        print("-" * 50)

    while r_norm > tol:
        for k in range(n_iter):
            tau_k = tau_0 / (1 + rho_0 * v[k])

            x = x - tau_k * r
            r = A @ x - b
            r_norm = np.linalg.norm(r)

            if log:
                total_iter += 1
                print(f"Приближение k = {total_iter}")
                print("-" * 50)
                print(f"Вектор x: {x}")
                print(f"Норма невязки: {r_norm}")
                print("-" * 50)
            
            if r_norm <= tol:
                break

    return x


def compute_parameters(A, n):
    """
    Вычисление параметров

    tau_0 = 2 / (lambda_min + lambda_max)
    rho_0 = (1 - eta) / (1 + eta)
    v_k = cos((2k-1)*pi / (2n)), k = 1, 2, ..., n
    """
    A_copy = A.copy()
    eigenvalues, _ = rotation_with_barriers(A_copy)

    lambda_min = np.min(eigenvalues)
    lambda_max = np.max(eigenvalues)

    eta = lambda_min / lambda_max

    tau_0 = 2.0 / (lambda_min + lambda_max)
    rho_0 = (1.0 - eta) / (1.0 + eta)

    k = np.arange(1, n + 1)
    v = np.cos((2 * k - 1) * np.pi / (2 * n))

    return tau_0, rho_0, v