import numpy as np

def check_singular(A, tol=1e-10):
    """Проверка вырожденности матрицы"""
    det = np.linalg.det(A)
    return abs(det) < tol

def check_residual(A, b, x):
    """Проверка невязки решения"""
    r = A @ x - b
    return np.linalg.norm(r)