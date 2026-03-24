import numpy as np

def check_singular(A, tol=1e-10):
    """Проверка вырожденности матрицы с помощью числа обусловленности"""
    try:
        cond_number = np.linalg.cond(A)
        
        if np.isinf(cond_number) or cond_number > 1/tol:
            print(f"(!) Матрица вырождена")
            return True
            
        return False
        
    except Exception as e:
        print(f"(!) Ошибка при проверке матрицы: {e}")
        return True

def check_residual(A, b, x):
    """Проверка невязки решения"""
    r = A @ x - b
    return np.linalg.norm(r)

def print_system(A, b):
    print("\nСистема:")
    print(A)
    print("\nВектор b:")
    print(b)

def execute_method(A, b, method):
    A_copy = A.copy()
    b_copy = b.copy()
    x = method(A_copy, b_copy)
    r = check_residual(A, b, x)

    print("\nРешение:")
    print(x)
    print("\nНевязка:")
    print(r)

    input("\nПродолжить...")