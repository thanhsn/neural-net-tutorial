import numpy as np
import sys
import matplotlib.pyplot as plt

N_INPUT = 2
N_HIDDEN = 4
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
    W0 = np.random.rand(N_HIDDEN, N_INPUT)
    W1 = np.random.rand(N_OUTPUT, N_HIDDEN)
    B0 = np.random.rand(N_HIDDEN, 1)
    B1 = np.random.rand(N_OUTPUT, 1)
    losses = []
    for t in xrange(N_ITERS):
        loss = 0.
        GW0 = np.zeros((N_HIDDEN, N_INPUT))
        GW1 = np.zeros((N_OUTPUT, N_HIDDEN))
        GB0 = np.zeros((N_HIDDEN, 1))
        GB1 = np.zeros((N_OUTPUT, 1))
        print "ITERATION", t
        for i in xrange(N_TRAIN):
            # Forward feeding
            z_1 = np.dot(W0, inp[:, i]) + B0[0]
            x_1 = sigmoid(z_1)
            z_output = np.dot(W1, x_1) + B1[0]
            x_output = sigmoid(z_output)
            p = x_output[0]
            y = ans[i]
            # print inp[0, i], "xor", inp[1, i], "=", y, ", Predict =", p 
            diff = p - y;
            loss += 0.5 * diff * diff
            # Back propagation
            delta_output = diff * (1 - p) * p
            GW1 += delta_output * np.transpose(x_1)
            GB1 += delta_output
            f_prime_z_1 = (1 - x_1) * x_1
            delta_1 = f_prime_z_1.reshape((N_HIDDEN, 1)) * (np.transpose(W1) * delta_output)
            GW0 += delta_1 * np.transpose(inp[:, i])
            GB0 += delta_1
        W1 -= ALPHA * GW1
        B1 -= ALPHA * GB1
        W0 -= ALPHA * GW0
        B0 -= ALPHA * GB0
        print "Loss at iteration", t, "=", loss
        losses.append(loss)
    print "W0 = ", W0
    print "B0 = ", B0
    print "W1 = ", W1
    print "B1 = ", B1
    plt.plot(range(N_ITERS), losses, 'ro')
    plt.ylim([0, 1])
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.show()


np.random.seed(100)
createInput()
train()

