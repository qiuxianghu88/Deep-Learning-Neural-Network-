import os
import sys
import torch
import torch.nn as nn
import random
from torch.autograd import Variable
from os.path import join as join_path
import torch.nn.functional as F

# make fake data
max_fid = 123
num_epochs = 10
data_path = './data/a8a'
batch_size = 1000


def emb_sum(embeds, weights=None):
    embeds_sum = (embeds * weights.unsqueeze(2).expand_as(embeds)).sum(1).squeeze(1)
    return embeds_sum


class Net(torch.nn.Module):
    def __init__(self, n_embed, n_feature, n_hidden, n_output):
        super(Net, self).__init__()

        self.embedding = nn.Embedding(max_fid + 1, n_feature)
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, fea_ids, fea_weights):
        embeds = self.embedding(fea_ids)
        embeds = emb_sum(embeds, fea_weights)
        embeds = nn.functional.tanh(embeds)
        hidden = self.hidden(embeds)
        output = self.out(hidden)
        return output


def read_and_shuffle(filepath, shuffle=True):
    lines = []
    with open(filepath, 'r') as fd:
        for line in fd:
            lines.append(line.strip())

    if shuffle:
        random.shuffle(lines)
    return lines


class DataLoader(object):
    def __init__(self, data_path, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        if os.path.isdir(data_path):
            self.files = [
                join_path(data_path, i) for i in os.listdir(data_path)
            ]
        else:
            self.files = [data_path]


    def parse(self, line_ite):
        for line in line_ite:
            it = line.strip().split(' ')
            label = int(it[0])

            f_id = []
            f_w = []
            for f_s in it[1:]:
                f, s = f_s.strip().split(':')
                f_id.append(int(f))
                f_w.append(float(s))
            f_id_len = len(f_id)

            output = []
            output.append(f_id)
            output.append(f_w)
            output.append(f_id_len)
            output.append(label)
            yield output


    def to_tensor(self, batch_data, sample_size):
        output = [None] * 3

        # id
        ids_buffer = torch.LongTensor(sample_size, max_fid)
        ids_buffer.fill_(1)
        output[0] = ids_buffer

        # weights
        weights_buffer = torch.zeros(sample_size, max_fid)
        output[1] = weights_buffer

        # label
        label_buffer = torch.LongTensor(sample_size)
        output[2] = label_buffer

        for sample_id, sample in enumerate(batch_data):
            f_id, f_w, f_id_len, label = sample
            output[0][sample_id, 0:f_id_len]  = torch.LongTensor(f_id)
            output[1][sample_id, 0:f_id_len]  = torch.FloatTensor(f_w)
            output[2][sample_id]  = label

        return tuple(output)

    def batch(self):
        count = 0
        batch_buffer = []
        for f in self.files:
            for parse_res in self.parse(read_and_shuffle(f, self.shuffle)):
                count += 1
                batch_buffer.append(parse_res)
                if count == self.batch_size:
                    t_size = len(batch_buffer)
                    batch_tensor = self.to_tensor(batch_buffer, t_size)
                    yield (batch_tensor, t_size)
                    count = 0
                    batch_buffer = []

        if batch_buffer:
            t_size = len(batch_buffer)
            batch_tensor = self.to_tensor(batch_buffer, t_size)
            yield (batch_tensor, t_size)


dataloader = DataLoader(data_path, batch_size)
net = Net(n_embed=32, n_feature=32, n_hidden=10, n_output=2)     # define the network

for epoch in range(1, num_epochs + 1):
    for (batch_id, data_tuple) in enumerate(dataloader.batch(), 1):
        data = data_tuple[0]
        all_sample_size = data_tuple[1]
        fid, fweight, label = data
        fid, fweight, label = map(Variable, [fid, fweight, label])

        optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
        loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

        out = net(fid, fweight)

        optimizer.zero_grad()   # clear gradients for next train
        loss = loss_func(out, label)
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients

        if batch_id % 5 == 0:
            # plot and show learning process
            prediction = torch.max(F.softmax(out), 1)[1]
            pred_y = prediction.data.numpy().squeeze()
            target_y = label.data.numpy()
            accuracy = sum(pred_y == target_y)/float(all_sample_size)

            print "epoch: %s, batch_id: %s, acc: %s" % (epoch, batch_id, accuracy)

            checkpoint = {
                'model': net.state_dict(),
                'epoch': epoch,
                'batch': batch_id,
            }
            model_name = 'epoch_%s_batch_id_%s_acc_%s.chkpt' % (epoch, batch_id, accuracy)
            model_path = join_path('./model', model_name)
            torch.save(checkpoint, model_path)
