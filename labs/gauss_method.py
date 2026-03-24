import numpy as np


def gauss_method(A: np.ndarray , b: np.ndarray) -> np.ndarray:
    """Модификация метода Гаусса с поиском максимального элемента по всей матрице"""
    n = len(b)
    row_order = np.arange(n)
    col_order = np.arange(n)

    for k in range(n):
        max_index = np.argmax(np.abs(A[row_order[k:], :][:, col_order[k:]]))
        max_row, max_col = divmod(max_index, n - k)

        if max_row != k:
            row_order[[k, k + max_row]] = row_order[[k + max_row, k]]
        
        if max_col != k:
            col_order[[k, k + max_col]] = col_order[[k + max_col, k]]

        pivot = A[row_order[k], col_order[k]]
        for i in range(k+1, n):
            d = A[row_order[i], col_order[k]] / pivot
            A[row_order[i], col_order[k:]] -= d * A[row_order[k], col_order[k:]]
            b[row_order[i]] -= d * b[row_order[k]]

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = b[row_order[i]]
        for j in range(i+1, n):
            x[i] -= A[row_order[i], col_order[j]] * x[j]
        x[i] /= A[row_order[i], col_order[i]]

    x_original = np.zeros(n)
    for i, col in enumerate(col_order):
        x_original[col] = x[i]
        
    return x_original