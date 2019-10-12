import sys
import numpy as np
import math

embedding_weight = np.load('dump/embedding.weight')
hidden_weight = np.load('./dump/hidden.weight')
hidden_bias = np.load('./dump/hidden.bias')
out_weight = np.load('./dump/out.weight')
out_bias = np.load('./dump/out.bias')

emb_dim = embedding_weight.shape[1]

def calc_score(fids, fweights):
    embedding = np.zeros(emb_dim)

    for id, weight in zip(fids, fweights):
        embedding += weight * embedding_weight[id]

    embedding_tanh = np.tanh(embedding)

    hidden = np.sum(np.multiply(hidden_weight, embedding_tanh), axis=1) + hidden_bias
    out = np.sum(np.multiply(out_weight, hidden), axis=1) + out_bias

    return out


def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


for line in sys.stdin:
    ss = line.strip().split(' ')
    label = ss[0].strip()
    fids = []
    fweights = []
    for f_s in ss[1:]:
        f, s = f_s.strip().split(':')
        fids.append(int(f))
        fweights.append(float(s))

    pred_label = softmax(calc_score(fids, fweights))[1]
    print label, pred_label
