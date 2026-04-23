import numpy as np


def generate_random_system(n: int, 
                    d: float = 10.0,
                    symmetrical: bool = False,
                    positive_definite: bool = False,
                    diagonally_dominant: bool = False,
                    dominance_factor: float = 1.1) -> tuple[np.ndarray, np.ndarray]:
    """
    Генерация системы линейных уравнений Ax = b с заданными свойствами
    
    Аргументы:
        n : int - Размер матрицы
        d : float - Диапазон значений [-d, d] для элементов матрицы и вектора
        symmetrical : bool - Сделать матрицу симметричной
        positive_definite : bool - Сделать матрицу положительно определённой 
        diagonally_dominant : bool - Обеспечить строгое диагональное преобладание
        dominance_factor : float - Коэффициент усиления диагонали (>1 для строгого преобладания)
    """
    
    b = np.random.uniform(-d, d, n)
    
    if positive_definite:
        M = np.random.uniform(-d, d, (n, n))
        A = M.T @ M + np.eye(n) * 1e-6
        
        if not symmetrical:
            skew = np.random.uniform(-d/2, d/2, (n, n))
            skew = skew - skew.T
            A = A + skew
            
    else:
        A = np.random.uniform(-d, d, (n, n))
        
        if symmetrical:
            A = np.triu(A) + np.triu(A, 1).T
    
    if diagonally_dominant:
        off_diag = A - np.diag(np.diag(A))
        row_sums = np.sum(np.abs(off_diag), axis=1)
        diag = row_sums * dominance_factor + np.random.uniform(0.1, 1.0, n)
        np.fill_diagonal(A, diag)
    
    scale = np.max(np.abs(A))
    if scale > d:
        A = A * (d / scale)

    return A, b


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