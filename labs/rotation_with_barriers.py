import numpy as np


def rotation_with_barriers(A, p=6):
    """Метод вращений с преградами для нахождения собственных значений и собственных векторов симметричной матрицы"""
    n = A.shape[0]
    V = np.eye(n)
    sigma = None
    
    for k in range(1, p + 1):
        sigma = compute_barrier(A, k)
        
        while True:
            i, j = find_max_offdiag(A)
            max_val = A[i, j]
            
            if abs(max_val) <= sigma:
                break
            
            c, s = compute_rotation_params(A, i, j)

            apply_rotation(A, i, j, c, s)
            update_eigenvectors(V, i, j, c, s)
    
    eigenvalues = np.diag(A)
    
    return eigenvalues, V


def compute_barrier(A, k):
    """Вычисление преграды по формуле: sigma_k = sqrt(max|a_{ii}|) * 10^(-k)"""
    max_diag = np.max(np.abs(np.diag(A)))
    sigma = np.sqrt(max_diag) * 10 ** (-k)
    return sigma


def find_max_offdiag(A):
    """Поиск максимального по модулю внедиагонального элемента"""
    B = A.copy()
    np.fill_diagonal(B, 0)
    
    i_max, j_max = np.unravel_index(np.argmax(np.abs(B)), A.shape)
    
    return i_max, j_max


def compute_rotation_params(A, i, j):
    """Вычисление c и s для обнуления элемента A[i, j]"""
    a_ii = A[i, i]
    a_jj = A[j, j]
    a_ij = A[i, j]
    
    diff = a_ii - a_jj
    
    d = np.sqrt(diff**2 + 4 * a_ij**2)
    
    if d == 0:
        c = np.sqrt(2) / 2
        s = c
    else:
        c = np.sqrt(0.5 * (1 + abs(diff) / d))
        
        sign = np.sign(a_ij * diff)
        if sign == 0:
            sign = 1
        s = sign * np.sqrt(0.5 * (1 - abs(diff) / d))
    
    return c, s


def apply_rotation(A, i, j, c, s):
    """Применение вращения: A_new = T_{ij}^T * A * T_{ij}"""
    n = A.shape[0]
    
    ai = A[i, :].copy()
    aj = A[j, :].copy()
    
    mask = np.ones(n, dtype=bool)
    mask[[i, j]] = False
    
    A[i, mask] = c * ai[mask] + s * aj[mask]
    A[mask, i] = A[i, mask]
    
    A[j, mask] = -s * ai[mask] + c * aj[mask]
    A[mask, j] = A[j, mask]
    
    A[i, i] = c*c * ai[i] + 2*c*s * ai[j] + s*s * aj[j]
    A[j, j] = s*s * ai[i] - 2*c*s * ai[j] + c*c * aj[j]
    A[i, j] = A[j, i] = 0.0


def update_eigenvectors(V, i, j, c, s):
    """Обновление матрицы собственных векторов: V_new = V * T_{ij}"""
    col_i = V[:, i].copy()
    col_j = V[:, j].copy()
    
    V[:, i] = c * col_i + s * col_j
    V[:, j] = -s * col_i + c * col_j