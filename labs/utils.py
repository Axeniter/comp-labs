import numpy as np


def is_singular(A, tol=1e-10):
    """Проверка вырожденности матрицы с помощью числа обусловленности"""
    try:
        cond_number = np.linalg.cond(A)
        
        if np.isinf(cond_number) or cond_number > 1/tol:
            return True
            
        return False
        
    except Exception as e:
        return True
    

def is_symmetrical(A, tol=1e-10):
    """Проверка симметричности матрицы"""
    return np.allclose(A, A.T, rtol=tol, atol=tol)


def is_spd(A: np.ndarray):
    """Проверка симметричности и положительной определённости матрицы"""
    if not is_symmetrical(A):
        return False
    
    eigenvalues = np.linalg.eigvalsh(A)
    
    return np.all(eigenvalues > 0)


def is_positive_definite(A):
    return np.min(np.linalg.eigvalsh(A)) > 0


def check_residual(A, b, x):
    """Проверка невязки решения"""
    r = A @ x - b
    return np.linalg.norm(r)


def print_system(A, b):
    print("Система:")
    print(A)
    print("\nВектор b:")
    print(b)


def execute_method(A, b, method, *args, **kwargs):
    A_copy = A.copy()
    b_copy = b.copy()
    x = method(A_copy, b_copy, *args, **kwargs)
    r = check_residual(A, b, x)

    print("\nРешение:")
    print(x)
    print("\nНевязка:")
    print(r)

    input("\nПродолжить...")


def execute_eigen_method(A, method, *args, **kwargs):
    A_copy = A.copy()
    eigenvalues, eigenvectors = method(A_copy, *args, **kwargs)
    eigenvalues, eigenvectors = sort_eigenpairs(eigenvalues, eigenvectors)
    r = check_eigen_residual(A, eigenvalues, eigenvectors)
    
    print("\nСобственные значения:")
    print(eigenvalues)
    print("\nСобственные векторы:")
    print(eigenvectors)
    print("\nНевязки:")
    print(r)
    
    input("\nПродолжить...")


def execute_partial_eigen_method(A, method, *args, **kwargs):
    A_copy = A.copy()
    eigenvalue, eigenvector = method(A_copy, *args, **kwargs)
    r = np.linalg.norm(A @ eigenvector - eigenvalue * eigenvector)
    
    print("\nСобственное значение:")
    print(eigenvalue)
    print("\nСобственный вектор:")
    print(eigenvector)
    print("\nНевязка:")
    print(r)
    
    input("\nПродолжить...")


def check_eigen_residual(A, eigenvalues, eigenvectors):
    n = len(eigenvalues)
    r = np.zeros(n)
    for i in range(n):
        r[i] = np.linalg.norm(A @ eigenvectors[:, i] - eigenvalues[i] * eigenvectors[:, i])
    return r


def sort_eigenpairs(eigenvalues, eigenvectors):
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]