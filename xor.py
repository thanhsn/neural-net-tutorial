import numpy as np
import sys
import matplotlib.pyplot as plt

N_INPUT = 2
N_OUTPUT = 1
N_TRAIN = 4
ALPHA = 0.1

N_ITERS = int(sys.argv[1])


inp = np.zeros((N_INPUT, N_TRAIN), dtype=int)
ans = np.zeros(N_TRAIN)

def createInput():
    for i in xrange(N_TRAIN):
        inp[0, i] = i % 2
        inp[1, i] = i / 2
        ans[i] = inp[0, i] ^ inp[1, i]

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def train():
    W = np.random.rand(N_OUTPUT, N_INPUT)
    B = np.random.rand(N_OUTPUT)
    losses = []
    for t in xrange(N_ITERS):
        loss = 0.
        GW = np.zeros((N_OUTPUT, N_INPUT))
        GB = np.zeros(N_OUTPUT)
        print "ITERATION", t
        for i in xrange(N_TRAIN):
            # Forward feeding
            z_output = np.dot(W, inp[:, i]) + B
            x_output = sigmoid(z_output)
            p = x_output[0]
            y = ans[i]
            # print inp[0, i], "xor", inp[1, i], "=", y, ", Predict =", p
            diff = p - y;
            loss += 0.5 * diff * diff
            # Back propagation
            delta_output = diff * (1 - p) * p
            GW += delta_output * np.transpose(inp[:, i])
            GB += delta_output
        W -= ALPHA * GW
        B -= ALPHA * GB
        print "Loss at iteration", t, "=", loss
        losses.append(loss)
    print "W = ", W
    print "B = ", B
    plt.plot(range(N_ITERS), losses, 'ro')
    plt.ylim([0, 1])
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.show()


np.random.seed(100)
createInput()
train()

