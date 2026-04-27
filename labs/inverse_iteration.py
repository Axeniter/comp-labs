import numpy as np
    

def inverse_iteration(A: np.ndarray, tol=1e-10, max_iter=1000, log=True):
    """Метод обратной итерации для нахождения наименьшего по модулю собственного значения матрицы"""
    n = A.shape[0]
    
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    for i in range(n):
        U[i, i:] = A[i, i:] - np.dot(L[i, :i], U[:i, i:])
        L[i:, i] = (A[i:, i] - np.dot(L[i:, :i], U[:i, i])) / U[i, i]
    
    if log:
        print("LU-разложение")
        print("-"*50)
        print("Матрица L:")
        print(L)
        print("\nМатрица U:")
        print(U)
        print("-"*50)
    
    def solve_with_lu(b):
        """Решение системы с уже разложенной матрицей"""
        y = np.zeros(n)
        for i in range(n):
            y[i] = b[i] - np.dot(L[i, :i], y[:i])
        
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
        
        return x
    
    x = np.random.rand(n)
    alpha = np.max(np.abs(x))
    alpha_prev = 0.0
    
    for k in range(1, max_iter + 1):
        b = x / alpha
        x = solve_with_lu(b)
        alpha = np.max(np.abs(x))
        
        if k > 1:
            lambda_current = 1.0 / alpha
            lambda_prev = 1.0 / alpha_prev
            
            if log:
                print(f"Шаг k = {k}")
                print("-"*50)
                print("Собственное значение:")
                print(lambda_current)
                print("\nСобственный вектор:")
                print(x)
                print("-"*50)
            
            if abs(lambda_current - lambda_prev) < tol:
                break
        
        alpha_prev = alpha
    
    return 1.0 / alpha, x