import numpy as np


def generate_random_vector(n: int, d: float = 10.0) -> np.ndarray:
    """Генерация случайного вектора b размера n со значениями в диапазоне [-d, d]"""
    return np.random.uniform(-d, d, n)


def generate_random_matrix(n: int, d: float = 10.0) -> np.ndarray:
    """Генерация случайной матрицы nxn со значениями в [-d, d]"""
    return np.random.uniform(-d, d, (n, n))


def generate_symmetric_matrix(n: int, d: float = 10.0) -> np.ndarray:
    """Генерация симметричной матрицы nxn со значениями в [-d, d]"""
    A = generate_random_matrix(n, d)
    return np.triu(A) + np.triu(A, 1).T


def generate_spd_matrix(n: int, d: float = 10.0, cond_number: float = 100.0) -> np.ndarray:
    """
    Генерация симметричной положительно определённой (SPD) матрицы nxn 
    со значениями в диапазоне [-d, d] и контролируемым числом обусловленности
    """
    M = np.random.randn(n, n)
    Q, _ = np.linalg.qr(M)
    
    eigenvalues = np.logspace(0, np.log10(cond_number), n)
    
    A = Q @ np.diag(eigenvalues) @ Q.T
    A = (A + A.T) / 2
    
    scale = np.max(np.abs(A))
    A = A * (d / scale)
    
    return A


def generate_diagonally_dominant_matrix(n: int, d: float = 10.0, dominance_factor: float = 1.1) -> np.ndarray:
    """Генерация матрицы nxn со строгим диагональным преобладанием в диапазоне [-d, d]"""
    A = generate_random_matrix(n, d)
    
    off_diag = A - np.diag(np.diag(A))
    row_sums = np.sum(np.abs(off_diag), axis=1)
    diag = row_sums * dominance_factor + np.random.uniform(0.1, 1.0, n)
    np.fill_diagonal(A, diag)
    
    scale = np.max(np.abs(A))
    A = A * (d / scale)

    return A


def create_input_system(n):
    """Ручное создание системы"""
    A = np.zeros((n,n))
    b = np.zeros(n)

    for i in range(n):
        print(f"\nУравнение {i + 1}:")
        for j in range(n):
            while True:
                try:
                    value = float(input(f"  a[{i+1}][{j+1}] = "))
                    A[i, j] = value
                    break
                except ValueError:
                    print("(!) Введите число")
        
        while True:
            try:
                value = float(input(f"  b[{i+1}] = "))
                b[i] = value
                break
            except ValueError:
                print("(!) Введите число")
    
    return A, b