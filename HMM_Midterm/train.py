from dspBox import str2ndar
from forward_backward import forward, backward
from viterbi import viterbi
import numpy as np

# validation
f = open('Obervations.txt','r')
vas = list()
for line in f.readlines():
    line = line.strip()
    vas.append(line)
f.close()

probs = np.zeros((30, 3))

for m in range(3):
    # init model
    a = np.array([[.3, .3, .4], [.3, .3, .4], [.4, .4, .2]])
    b = np.array([[.4, .4, .2], [.3, .3, .4], [.3, .3, .4]])
    pi = np.array([.3, .3, .4])
    
    # training
    for it in range(3):
        for ns in range(100):
            # open training data
            f = open('training/model_%s_training.txt' % (m + 1),'r')
            obs = list()
            for line in f.readlines():
                line = line.strip()
                obs.append(line)
            f.close()     

            ob = str2ndar(obs[ns])
            N = a.shape[0]
            T = len(ob)

            # update a
            for i in range(N):
                for j in range(N):
                    nu = 0
                    for t in range(T - 1):
                        alpha = forward(ob, a, b, pi); beta = backward(ob, a, b, pi)
                        nu += alpha[t][i] * b[j][ob[t + 1]] * beta[t + 1][j] / sum([alpha[t][n] * beta[t][n] for n in range(N)])
                    de = 0
                    for t in range(T - 1):
                        for jj in range(N):
                            alpha = forward(ob, a, b, pi); beta = backward(ob, a, b, pi)
                            de += alpha[t][i] * b[jj][ob[t + 1]] * beta[t + 1][jj] / sum([alpha[t][n] * beta[t][n] for n in range(N)])
                    a[i][j] = nu / de
            print(a, '\n')
            
            # update b
            for i in range(N):
                for j in range(N):
                    nu = 0
                    for t in range(T):
                        if j == ob[t]:
                            alpha = forward(ob, a, b, pi); beta = backward(ob, a, b, pi)
                            nu += alpha[t][j] * beta[t][j] / sum([alpha[t][n] * beta[t][n] for n in range(N)])
                    de = 0
                    for t in range(T):
                        alpha = forward(ob, a, b, pi); beta = backward(ob, a, b, pi)
                        de += alpha[t][j] * beta[t][j] / sum([alpha[t][n] * beta[t][n] for n in range(N)])
                    b[i][j] = nu / de
            print(b, '\n')

            # update pi
            for i in range(N):
                lpha = forward(ob, a, b, pi); beta = backward(ob, a, b, pi)
                pi[i] = alpha[0][i] * beta[0][i] / sum([alpha[0][n] * beta[0][n] for n in range(N)])
            print(pi, '\n\n----------------------------------\n')
    
    # output model
    fp = open("model_%s.txt" % (m + 1), "a")
    fp.seek(0)
    fp.truncate()
    fp.writelines('a = \n%s\n\n' % (a))
    fp.writelines('b = \n%s\n\n' % (b))
    fp.writelines('pi = \n%s\n\n' % (pi))
    fp.close()

    # save P(validation|model_m)
    for ns in range(30):
        va = str2ndar(vas[ns])
        probs[ns][m] = viterbi(va, a, b, pi)

# output answer
fp = open("Observations_Ans.txt", "a")
fp.seek(0)
fp.truncate()
for ns in range(30):
    max_idx = np.argmax(probs[ns])
    fp.writelines('model %s\n' % (max_idx + 1))
fp.close()
print('Done.')
