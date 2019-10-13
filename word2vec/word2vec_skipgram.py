import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from torch.nn.parameter import Parameter

from collections import Counter
import numpy as np
import random
import math

import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

# this parameter decides whehter this code runs on GPU or CPU
# the speed of CPU is slower than GPU
USE_CUDA = torch.cuda.is_available()

# 为了保证复现，random seed固定
random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
if USE_CUDA:
    torch.cuda.manual_seed(53113)

# 超参数
# 一个正样本对应100个负样本
K = 100  # number of negative samples
#窗口大小
C = 3  # nearby words threshold
NUM_EPOCHS = 2  # The number of epochs of training
#取3万个高频单词，不需要所有词汇都学习
MAX_VOCAB_SIZE = 30000  # the vocabulary size
BATCH_SIZE = 128  # the batch size
LEARNING_RATE = 0.2  # the initial learning rate
EMBEDDING_SIZE = 100

LOG_FILE = "./log/word-embedding.log"

# 分词
def word_tokenize(text):
    return text.split()

with open("./data/text8", "r") as fin:
    text = fin.read()

text = [w for w in word_tokenize(text.lower())]

vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))
# 所有低频词用unk表示
vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))
#将单词id化
idx_to_word = [word for word in vocab.keys()]
word_to_idx = {word: i for i, word in enumerate(idx_to_word)}
# 计算单词词频，为后面负采样做准备
word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
word_freqs = word_counts / np.sum(word_counts)
word_freqs = word_freqs ** (3. / 4.)
word_freqs = word_freqs / np.sum(word_freqs)  # 用来做 negative sampling
VOCAB_SIZE = len(idx_to_word)

print(VOCAB_SIZE)

class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word_to_idx, idx_to_word, word_freqs, word_counts):
        ''' text: a list of words, all text from the training dataset
            word_to_idx: the dictionary from word to idx
            idx_to_word: idx to word mapping
            word_freq: the frequency of each word
            word_counts: the word counts
        '''
        super(WordEmbeddingDataset, self).__init__()
        self.text_encoded = [word_to_idx.get(t, VOCAB_SIZE - 1) for t in text]
        self.text_encoded = torch.Tensor(self.text_encoded).long()
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)

    def __len__(self):
        ''' 返回整个数据集（所有单词）的长度
        '''
        return len(self.text_encoded)

    def __getitem__(self, idx):
        ''' 这个function返回以下数据用于训练
            - 中心词
            - 这个单词附近的(positive)单词
            - 随机采样的K个单词作为negative sample
        '''
        center_word = self.text_encoded[idx]
        # C是窗口大小
        pos_indices = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))
        # 对i求余数的原因是最左边可能是负数，因为如果中心词是第一个的话
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]
        # 对text编码，即往tensor去映射
        pos_words = self.text_encoded[pos_indices]
        # 根据单词频率随机取负样本，每个正样本取K个负样本
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)

        return center_word, pos_words, neg_words

dataset = WordEmbeddingDataset(text, word_to_idx, idx_to_word, word_freqs, word_counts)
dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        ''' 初始化输出和输出embedding
        '''
        super(EmbeddingModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        initrange = 0.5 / self.embed_size
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.out_embed.weight.data.uniform_(-initrange, initrange)

        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.in_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_labels, pos_labels, neg_labels):
        '''
        input_labels: 中心词, [batch_size]
        pos_labels: 中心词周围 context window 出现过的单词 [batch_size * (window_size * 2)]
        neg_labelss: 中心词周围没有出现过的单词，从 negative sampling 得到 [batch_size, (window_size * 2 * K)]

        return: loss
        '''

        input_embedding = self.in_embed(input_labels)  # B * embed_size
        pos_embedding = self.out_embed(pos_labels)  # B * (2*C) * embed_size
        neg_embedding = self.out_embed(neg_labels)  # B * (2*C * K) * embed_size
        print("input_embedding size:", input_embedding.size())
        print("pos_embedding size:", pos_embedding.size())
        print("neg_embedding size:", neg_embedding.size())
        # 通过unsqueeze强行增加了一个第三维度（0是行，1是列，2是第三个维度），维度相同，后面才能进行求和计算
        log_pos = torch.bmm(pos_embedding, input_embedding.unsqueeze(2)).squeeze()  # B * (2*C)
        log_neg = torch.bmm(neg_embedding, -input_embedding.unsqueeze(2)).squeeze()  # B * (2*C*K)
        print("log_pos size:", log_pos.size())
        print("log_neg size:", log_neg.size())
        log_pos = F.logsigmoid(log_pos).sum(1)
        log_neg = F.logsigmoid(log_neg).sum(1)

        loss = log_pos + log_neg
        print("log_pos size:", log_pos.size())
        print("log_neg size:", log_neg.size())
        print("loss size:", loss.size())
        sys.exit()

        # 计算负对数释然
        return -loss

    def input_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()

model = EmbeddingModel(VOCAB_SIZE, EMBEDDING_SIZE)
if USE_CUDA:
    model = model.cuda()

def evaluate(filename, embedding_weights):
    if filename.endswith(".csv"):
        data = pd.read_csv(filename, sep=",")
    else:
        data = pd.read_csv(filename, sep="\t")
    human_similarity = []
    model_similarity = []
    for i in data.iloc[:, 0:2].index:
        word1, word2 = data.iloc[i, 0], data.iloc[i, 1]
        if word1 not in word_to_idx or word2 not in word_to_idx:
            continue
        else:
            word1_idx, word2_idx = word_to_idx[word1], word_to_idx[word2]
            word1_embed, word2_embed = embedding_weights[[word1_idx]], embedding_weights[[word2_idx]]
            model_similarity.append(float(sklearn.metrics.pairwise.cosine_similarity(word1_embed, word2_embed)))
            human_similarity.append(float(data.iloc[i, 2]))

    return scipy.stats.spearmanr(human_similarity, model_similarity)# , model_similarity

def find_nearest(word):
    index = word_to_idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return [idx_to_word[i] for i in cos_dis.argsort()[:10]]


if __name__ == '__main__':

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    for e in range(NUM_EPOCHS):
        for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):

            # print(input_labels.size())    # [128]
            # print(pos_labels.size())      # [128, 6]
            # print(neg_labels.size())      # [128, 600]

            input_labels = input_labels.long()
            pos_labels = pos_labels.long()
            neg_labels = neg_labels.long()
            if USE_CUDA:
                input_labels = input_labels.cuda()
                pos_labels = pos_labels.cuda()
                neg_labels = neg_labels.cuda()

            optimizer.zero_grad()
            loss = model(input_labels, pos_labels, neg_labels).mean()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                with open(LOG_FILE, "a") as fout:
                    fout.write("epoch: {}, iter: {}, loss: {}\n".format(e, i, loss.item()))
                    print("epoch: {}, iter: {}, loss: {}".format(e, i, loss.item()))
        # 保存embedding
        embedding_weights = model.input_embeddings()
        torch.save(model.state_dict(), "embedding-{}.th".format(EMBEDDING_SIZE))