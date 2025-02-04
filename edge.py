import numpy as np
import math
from scipy.special import *
from sklearn.neighbors import NearestNeighbors
import torch 

def gen_W(X, Y):
    np.random.seed(3334)
    N = X.shape[0]
    dim_X, dim_Y = X.shape[1], Y.shape[1]
    
    kx, ky = 2, 2
    rx, ry = 10, 10
    
    # std_X = np.array([np.std(X[:, [i]]) for i in range(dim_X)]).reshape(dim_X, 1)
    std_X = np.array([np.std(X[:, [i]].cpu().numpy()) for i in range(dim_X)]).reshape(dim_X, 1)

    std_Y = np.array([np.std(Y[:, [i]].cpu().numpy()) for i in range(dim_Y)]).reshape(dim_Y, 1)
    
    d_X_shrink = min(dim_X, math.floor(math.log(N / rx, kx)))
    d_Y_shrink = min(dim_Y, math.floor(math.log(N / ry, ky)))
    
    std_X_mat = np.tile(std_X, (1, d_X_shrink))
    std_Y_mat = np.tile(std_Y, (1, d_Y_shrink))
    
    std_X_mat[std_X_mat < 0.0001] = 1
    std_Y_mat[std_Y_mat < 0.0001] = 1
    
    sigma_X = 1.0 / (std_X_mat * np.sqrt(dim_X))
    sigma_Y = 1.0 / (std_Y_mat * np.sqrt(dim_Y))
    
    W = np.random.normal(0, sigma_X, (dim_X, d_X_shrink))
    V = np.random.normal(0, sigma_Y, (dim_Y, d_Y_shrink))
    
    return W, V

def find_knn(A, d):
    r = 500
    N = A.shape[0]
    k = math.floor(0.43 * N**(2/3 + 0.17 * (d / (d + 1))) * math.exp(-1.0 / max(10000, d**4)))
    
    T = np.random.choice(A.reshape(-1,), size=r).reshape(-1, 1)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(A.reshape(-1, 1))
    distances, _ = nbrs.kneighbors(T)
    return np.mean(distances[:, -1])

def gen_eps(XW, YV):
    eps_X = np.array([find_knn(XW[:, [i]], XW.shape[1]) for i in range(XW.shape[1])]) + 0.0001
    eps_Y = np.array([find_knn(YV[:, [i]], YV.shape[1]) for i in range(YV.shape[1])]) + 0.0001
    return eps_X, eps_Y

def H1(XW, b, eps):
    X_te = (XW + b) / eps
    return tuple(np.floor(X_te).tolist())

def Hash(XW, YV, eps_X, eps_Y, b_X, b_Y):
    N = XW.shape[0]
    CX, CY, CXY = {}, {}, {}

    # print(XW.shape)  
    # print(YV.shape)  
    
    for i in range(N):
        
        X_l, Y_l = H1(XW[i], b_X, eps_X), H1(YV[i], b_Y, eps_Y)
        CX.setdefault(X_l, []).append(i)
        CY.setdefault(Y_l, []).append(i)
        CXY.setdefault((X_l, Y_l), []).append(i)
    
    return CX, CY, CXY

def Compute_MI(XW, YV, U, eps_X, eps_Y, b_X, b_Y):
    CX, CY, CXY = Hash(XW, YV, eps_X, eps_Y, b_X, b_Y)
    I, N_c = 0, 0
    
    for e in CXY.keys():
        Ni, Mj, Nij = len(CX[e[0]]), len(CY[e[1]]), len(CXY[e])
        I += Nij * max(min(math.log(Nij * len(XW) / (Ni * Mj), 2), U), 0.001)

        N_c += Nij
    
    return I / N_c

def EDGE(X, Y, U=10, gamma=[1, 1], epsilon=[0, 0], hashing='p-stable', L_ensemble=5):
    gamma, epsilon = np.array(gamma), np.array(epsilon)
    if X.ndim == 1: X = X.reshape((-1, 1))

    if Y.ndim == 1:
        Y = Y.reshape((-1, 1))
    
    if hashing == 'p-stable':
        W, V = gen_W(X, Y)
        XW, YV = np.dot(X, W), np.dot(Y, V)
    else:
        XW, YV = X, Y
    
    if epsilon[0] == 0:
        eps_X_temp, eps_Y_temp = gen_eps(XW, YV)
        eps_X0, eps_Y0 = eps_X_temp * 3 * XW.shape[1] * gamma[0], eps_Y_temp * 3 * YV.shape[1] * gamma[1]
    else:
        eps_X0, eps_Y0 = np.ones(XW.shape[1]) * epsilon[0], np.ones(YV.shape[1]) * epsilon[1]
    
    T = np.ones(L_ensemble)
    I_vec = np.zeros(L_ensemble)
    
    for j in range(L_ensemble):
        eps_X, eps_Y = eps_X0 * T[j], eps_Y0 * T[j]
        b_X, b_Y = np.linspace(0, 1, L_ensemble, endpoint=False)[j] * eps_X, np.linspace(0, 1, L_ensemble, endpoint=False)[j] * eps_Y
        I_vec[j] = Compute_MI(XW, YV, U, eps_X, eps_Y, b_X, b_Y)
    
    return np.median(I_vec)

def estimate_mi(X, Y, U=10, gamma=[1, 1], epsilon=[0, 0], hashing='p-stable'):
    return EDGE(X, Y, U=U, gamma=gamma, epsilon=epsilon, hashing=hashing)


