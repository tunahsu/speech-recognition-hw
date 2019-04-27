from dspBox import str2ndar
import numpy as np

def forwardprob(ob, a, b, pi):
    probs = np.zeros((len(ob), len(pi)))
    
    for i in range(len(ob)):
        for j in range(len(pi)):
            if i == 0:
                probs[i][j] = pi[j] * b[j, ob[i]]
            else:
                for k in range(len(pi)):
                    probs[i][j] += probs[i - 1][k] * a[k, j]
                probs[i][j] *= b[j, ob[i]]
        if i == (len(ob) - 1):
            prob = sum(probs[i])
    return prob

def backwardprob(ob, a, b, pi):
    probs = np.zeros((len(ob), len(pi)))
    
    for i in range((len(ob) -1), -1, -1):
        for j in range(len(pi)):
            if i == (len(ob) -1):
                probs[i][j] = 1
            else:
                for k in range(len(pi)):
                    probs[i][j] += a[j, k] * b[k, ob[i + 1]] * probs[i + 1][k]
        if i == 0:
            prob = sum(pi[l] * b[l, ob[i]] * probs[i][l] for l in range(len(pi)))
    return prob

f1 = open('obser1.txt', 'r')
ob1 = str2ndar(f1.read())
a1 = np.matrix('0.2, 0.7, 0.1; 0.1, 0.2, 0.7; 0.7, 0.1, 0.2')
b1 = np.matrix('0.5, 0.4, 0.1; 0.7, 0.2, 0.1; 0.7, 0.1, 0.2')
pi1 = np.array([0.7, 0.2, 0.1])

f2 = open('obser2.txt', 'r')
ob2 = str2ndar(f2.read())
a2 = np.matrix('0.7, 0.2, 0.1; 0.3, 0.6, 0.1; 0.1, 0.2, 0.7')
b2 = np.matrix('0.1, 0.8, 0.1; 0.2, 0.7, 0.1; 0.4, 0.5, 0.1')
pi2 = np.array([0.1, 0.7, 0.2])

f3 = open('obser3.txt', 'r')
ob3 = str2ndar(f3.read())
a3 = np.matrix('0.2, 0.7, 0.1; 0.6, 0.3, 0.1; 0.2, 0.7, 0.1')
b3 = np.matrix('0.1, 0.2, 0.7; 0.2, 0.2, 0.6; 0.3, 0.1, 0.6')
pi3 = np.array([0.2, 0.2, 0.6])

print('obser1')
print('model1 forward: {:.6e} | backword: {:.6e}'.format(forwardprob(ob1, a1, b1, pi1), backwardprob(ob1, a1, b1, pi1)))
print('model2 forward: {:.6e} | backword: {:.6e}'.format(forwardprob(ob1, a2, b2, pi2), backwardprob(ob1, a2, b2, pi2)))
print('model3 forward: {:.6e} | backword: {:.6e}'.format(forwardprob(ob1, a3, b3, pi3), backwardprob(ob1, a3, b3, pi3)))

print('obser2')
print('model1 forward: {:.6e} | backword: {:.6e}'.format(forwardprob(ob2, a1, b1, pi1), backwardprob(ob2, a1, b1, pi1)))
print('model2 forward: {:.6e} | backword: {:.6e}'.format(forwardprob(ob2, a2, b2, pi2), backwardprob(ob2, a2, b2, pi2)))
print('model3 forward: {:.6e} | backword: {:.6e}'.format(forwardprob(ob2, a3, b3, pi3), backwardprob(ob2, a3, b3, pi3)))

print('obser3')
print('model1 forward: {:.6e} | backword: {:.6e}'.format(forwardprob(ob3, a1, b1, pi1), backwardprob(ob3, a1, b1, pi1)))
print('model2 forward: {:.6e} | backword: {:.6e}'.format(forwardprob(ob3, a2, b2, pi2), backwardprob(ob3, a2, b2, pi2)))
print('model3 forward: {:.6e} | backword: {:.6e}'.format(forwardprob(ob3, a3, b3, pi3), backwardprob(ob3, a3, b3, pi3)))