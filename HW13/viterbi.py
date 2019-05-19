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
    return max(probs[i]), states

ob = [None] * 3; a = [None] * 3; b = [None] * 3; pi = [None] * 3

f1 = open('Observation_1.txt', 'r')
ob[0] = str2ndar(f1.read())
a[0] = np.matrix('0.2, 0.7, 0.1; 0.1, 0.2, 0.7; 0.7, 0.1, 0.2')
b[0] = np.matrix('0.5, 0.4, 0.1; 0.7, 0.2, 0.1; 0.7, 0.1, 0.2')
pi[0] = np.array([0.7, 0.2, 0.1])

f2 = open('Observation_2.txt', 'r')
ob[1] = str2ndar(f2.read())
a[1] = np.matrix('0.7, 0.2, 0.1; 0.3, 0.6, 0.1; 0.1, 0.2, 0.7')
b[1] = np.matrix('0.1, 0.8, 0.1; 0.2, 0.7, 0.1; 0.4, 0.5, 0.1')
pi[1] = np.array([0.1, 0.7, 0.2])

f3 = open('Observation_3.txt', 'r')
ob[2] = str2ndar(f3.read())
a[2] = np.matrix('0.2, 0.7, 0.1; 0.6, 0.3, 0.1; 0.2, 0.7, 0.1')
b[2] = np.matrix('0.1, 0.2, 0.7; 0.2, 0.2, 0.6; 0.3, 0.1, 0.6')
pi[2] = np.array([0.2, 0.2, 0.6])

for i in range(3):
    print('obser{}'.format(i + 1))
    for j in range(3):
        prob, states = viterbi(ob[i], a[j], b[j], pi[j])
        print('model_{} probability: {:.6e}'.format(j + 1, prob))
        print('viterbi max state sequence: {}'.format(states))
    print('\n')

