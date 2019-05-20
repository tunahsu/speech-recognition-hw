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
    # get training data
    f = open('training/model_%s_training.txt' % (m + 1), 'r')
    obs = list()
    for line in f.readlines():
        line = line.strip()
        obs.append(line)
    f.close()    

    # init model
    a = np.array([[.3, .3, .4], [.3, .3, .4], [.4, .4, .2]])
    b = np.array([[.4, .4, .2], [.3, .3, .4], [.3, .3, .4]])
    pi = np.array([.3, .3, .4])
    
    # training
    for it in range(10):
        for ns in range(100): 
            ob = str2ndar(obs[ns])
            N = a.shape[0]
            T = len(ob)
            alpha = forward(ob, a, b, pi); beta = backward(ob, a, b, pi)
            xi = np.zeros((T - 1, N, N))
            gammai = np.zeros((T, N))
            gammaj = np.zeros((T, N))

            # compute xi    
            for t in range(T - 1):
                de = sum([sum([alpha[t][j] * a[i][j] * b[j][ob[t + 1]] * beta[t + 1][j] for j in range(N)]) for i in range(N)])
                for i in range(N):
                    for j in range(N):
                        nu = alpha[t][i] * a[i][j] * b[j][ob[t + 1]] * beta[t + 1][j]
                        xi[t][i][j] = nu / de
            
            # compute gamma
            for t in range(T - 1):
                for n in range(N):
                    gammai[t][n] = sum([xi[t][n][nn] for nn in range(N)])
                    gammaj[t][n] = sum([xi[t][nn][n] for nn in range(N)])

            # update a
            for i in range(N):
                de = sum([gammai[t][i] for t in range(T - 1)])
                for j in range(N):
                    nu = sum([xi[t][i][j] for t in range(T - 1)])
                    a[i][j] = nu / de
            print(a, '\n')
            
            # update b
            for j in range(N):
                de = sum([gammaj[t][j] for t in range(T)])
                for k in range(N):
                    nu = sum([gammaj[t][j] for t in range(T) if k == ob[t]])
                    b[j][k] = nu / de
            print(b, '\n')

            # update pi
            for i in range(N):
                pi[i] = gammai[0][i]
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
