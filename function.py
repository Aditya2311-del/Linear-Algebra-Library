import numpy as np

def rref(A):
    M = A.astype(float).copy()
    rows, cols = M.shape
    lead = 0
    
    for r in range(rows):
        if lead >= cols:
            break
        i = r
        while abs(M[i, lead]) < 1e-10:
            i += 1
            if i == rows:
                i = r
                lead += 1
                if lead == cols:
                    return M
        M[[i, r]] = M[[r, i]]
        M[r] = M[r] / M[r, lead]
        for i in range(rows):
            if i != r:
                M[i] = M[i] - M[i, lead] * M[r]
        lead += 1
    
    return M


def gram_schmidt(V):
    V = np.array(V, dtype=float)
    n, m = V.shape
    Q = np.zeros((n, m))
    
    for i in range(m):
        q = V[:, i]
        for j in range(i):
            q = q - np.dot(Q[:, j], V[:, i]) * Q[:, j]
        q = q / np.linalg.norm(q)
        Q[:, i] = q
    
    return Q

def inverse(A):
    n = A.shape[0]
    Aug = np.hstack([A.astype(float), np.eye(n)])
    
    for i in range(n):
        pivot = np.argmax(np.abs(Aug[i:, i])) + i
        Aug[[i, pivot]] = Aug[[pivot, i]]
        Aug[i] = Aug[i] / Aug[i, i]
        for j in range(n):
            if i != j:
                Aug[j] = Aug[j] - Aug[j, i] * Aug[i]
    
    return Aug[:, n:]

def determinant(A):
    U = A.astype(float).copy()
    n = U.shape[0]
    swaps = 0
    
    for k in range(n - 1):
        pivot = np.argmax(np.abs(U[k:, k])) + k
        if pivot != k:
            U[[k, pivot]] = U[[pivot, k]]
            swaps += 1
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            U[i, k:] = U[i, k:] - factor * U[k, k:]
    
    det = (-1) ** swaps * np.prod(np.diag(U))
    return det

def lu_decomposition(A):
    n = A.shape[0]
    L = np.eye(n)
    U = A.astype(float).copy()
    
    for k in range(n - 1):
        pivot = np.argmax(np.abs(U[k:, k])) + k
        if pivot != k:
            U[[k, pivot]] = U[[pivot, k]]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] = U[i, k:] - factor * U[k, k:]
    
    return L, U
