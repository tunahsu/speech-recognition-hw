from dspBox import str2ndar
import numpy as np

def forward(ob, a, b, pi):
    N = a.shape[0]
    T = len(ob)
    probs = np.zeros((T, N))
    
    for i in range(T):
        for j in range(N):
            if i == 0:
                probs[i][j] = pi[j] * b[j, ob[i]]
            else:
                for k in range(N):
                    probs[i][j] += probs[i - 1][k] * a[k, j]
                probs[i][j] *= b[j, ob[i]]
        if i == (T - 1):
            prob = sum(probs[i])
    return probs

def backward(ob, a, b, pi):
    N = a.shape[0]
    T = len(ob)
    probs = np.zeros((T, N))
    
    for i in range(T - 1, -1, -1):
        for j in range(N):
            if i == (T -1):
                probs[i][j] = 1
            else:
                for k in range(N):
                    probs[i][j] += a[j, k] * b[k, ob[i + 1]] * probs[i + 1][k]
        if i == 0:
            prob = sum(pi[l] * b[l, ob[i]] * probs[i][l] for l in range(N))
    return probs