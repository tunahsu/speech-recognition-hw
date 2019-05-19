from dspBox import str2ndar
import numpy as np

def viterbi(ob, a, b, pi):
    probs = np.zeros((len(ob), len(pi)))
    probs_s = np.zeros((len(ob), len(pi)))
    states = np.zeros(len(ob), dtype=np.int8)
    for i in range(len(ob)):
        for j in range(len(pi)):
            if i == 0:
                probs[i][j] = pi[j] * b[j, ob[i]]
                probs_s[i][j] = pi[j] * b[j, ob[i]]
            else:
                probs[i][j] = max([probs[i - 1][k] * a[k, j] * b[j, ob[i]] for k in range(len(pi))])
                probs_s[i][j] = np.argmax([probs[i - 1][k] * a[k, j] * b[j, ob[i]] for k in range(len(pi))])
    states[i] = np.argmax(probs[i])
    for k in range(i - 1, -1, -1):
        states[k] = probs_s[k + 1][states[k + 1]]
    return max(probs[i])

