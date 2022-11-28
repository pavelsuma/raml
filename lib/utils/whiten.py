import os
import numpy as np
import torch


def whitenapply(X, m, P, dimensions=None):
    
    if not dimensions:
        dimensions = P.shape[0]
    
    X = np.dot(P[:dimensions, :], X-m)
    X = X / (np.linalg.norm(X, ord=2, axis=0, keepdims=True) + 1e-6)

    return X


def whitenapply_torch(X, m, P, dimensions=None):
    if not dimensions:
        dimensions = P.size(0)

    X = torch.mm(P[:dimensions, :], X - m)
    X = X / (torch.norm(X, dim=0, keepdim=True) + 1e-6)

    return X


def whitenapply_torch_chunked(X, m, P, dimensions=None):
    if not dimensions:
        dimensions = P.size(0)

    # done in chunks to fit in memory for the whole METU
    for i in range(10):
        low = int(i * (X.size(1) / 10))
        up = int((i+1) * (X.size(1) / 10))
        X[:dimensions, low:up] = torch.mm(P[:dimensions, :], X[:, low:up] - m)
        X[:dimensions, low:up] = X[:dimensions, low:up] / (torch.norm(X[:dimensions, low:up], dim=0, keepdim=True) + 1e-6)
    X = X[:dimensions, :]

    return X


def pcawhitenlearn(X):

    N = X.shape[1]

    # Learning PCA w/o annotations
    m = X.mean(axis=1, keepdims=True)
    Xc = X - m
    Xcov = np.dot(Xc, Xc.T)
    Xcov = (Xcov + Xcov.T) / (2*N)
    eigval, eigvec = np.linalg.eig(Xcov)
    order = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    P = np.dot(np.linalg.inv(np.sqrt(np.diag(eigval))), eigvec.T)
    
    return m, P


def pcawhitenlearn_torch(X, min_eig=-1):

    N = X.shape[1]

    # Learning PCA w/o annotations
    m = X.mean(axis=1, keepdims=True)
    Xc = X - m
    Xcov = torch.mm(Xc, Xc.T)
    Xcov = (Xcov + Xcov.T) / (2*N)
    eigval, eigvec = torch.symeig(Xcov, eigenvectors=True)
    order = torch.argsort(eigval, descending=True)
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    beta = eigval[min_eig]
    P = torch.mm(torch.inverse(torch.sqrt(torch.diag((1-beta)*eigval + beta))), eigvec.T)
    return m, P


def whitenlearn_torch(X, qidxs, pidxs):

    # Learning Lw w annotations
    # m = X.mean(axis=1, keepdims=True)
    m = X[:, qidxs].mean(axis=1, keepdims=True)
    df = X[:, qidxs] - X[:, pidxs]
    S = torch.mm(df, df.T) / df.shape[1]
    P = torch.inverse(cholesky_torch(S))
    df = torch.mm(P, X-m)
    D = torch.mm(df, df.T)
    eigval, eigvec = torch.symeig(D, eigenvectors=True)
    order = torch.argsort(eigval, descending=True)
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    P = torch.mm(eigvec.T, P)

    return m, P


def whitenlearn(X, qidxs, pidxs):

    # Learning Lw w annotations
    m = X[:, qidxs].mean(axis=1, keepdims=True)
    df = X[:, qidxs] - X[:, pidxs]
    S = np.dot(df, df.T) / df.shape[1]
    P = np.linalg.inv(cholesky(S))
    df = np.dot(P, X-m)
    D = np.dot(df, df.T)
    eigval, eigvec = np.linalg.eig(D)
    order = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    P = np.dot(eigvec.T, P)

    return m, P


def cholesky(S):
    # Cholesky decomposition
    # with adding a small value on the diagonal
    # until matrix is positive definite
    alpha = 0
    while 1:
        try:
            L = np.linalg.cholesky(S + alpha*np.eye(*S.shape))
            return L
        except:
            if alpha == 0:
                alpha = 1e-10
            else:
                alpha *= 10
            print(">>>> {}::cholesky: Matrix is not positive definite, adding {:.0e} on the diagonal"
                .format(os.path.basename(__file__), alpha))


def cholesky_torch(S):
    # Cholesky decomposition
    # with adding a small value on the diagonal
    # until matrix is positive definite
    alpha = 0
    while 1:
        try:
            L = torch.cholesky(S + alpha*np.eye(*S.shape))
            return L
        except:
            if alpha == 0:
                alpha = 1e-10
            else:
                alpha *= 10
            print(">>>> {}::cholesky: Matrix is not positive definite, adding {:.0e} on the diagonal"
                .format(os.path.basename(__file__), alpha))